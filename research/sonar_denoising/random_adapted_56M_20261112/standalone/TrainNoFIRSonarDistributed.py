"""Four-rank raw-input multi-source pretraining -> native adaptation.

No FIR-prepared field bank is read. Original typed observation windows receive
new perturbations; the raw reference is not certified clean. Rank ledgers record
actual source exposure. A global step is ONE actual optimizer update using one
sample per rank, not four fictitious updates. Evaluation never updates weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from NoFIRSonarWavelet import NoFIRSonarMWCNN, PROTOCOL
from NoFIRSonarData import Sources, example, plan_source, read, save, sha, json_text
from TrainNativeLocalSonarWavelet import fidelity_loss


@torch.inference_mode()
def validate(model, records, device, profile, phase, smoke):
    model.eval()
    roles = ('public_validation',) if phase == 'pretrain' else ('native_validation',)
    items = [r for r in records if r['role'] in roles]
    if smoke:
        items = items[:2]
    sources = Sources()
    sums = dict(sse=0., identity_sse=0., input_sse=0., count=0)
    for index, item in enumerate(items):
        data, core, _ = example(sources, item, 800000+index, phase, device, profile, validation=True)
        features, mask, base, target = data
        prediction = model(features, mask, item['domain'])[core].float()
        valid = mask[core]
        err = (base[core]+prediction-target[core])[valid].double()
        initial = (base[core]-target[core])[valid].double()
        identity_data, identity_core, _ = example(sources, item, 800000+index, phase,
            device, profile, validation=True, force_identity=True)
        rf, rm, _, _ = identity_data
        identity = model(rf, rm, item['domain'])[identity_core][valid].double()
        if not torch.isfinite(err).all() or not torch.isfinite(identity).all():
            raise ValueError('nonfinite predeclared validation')
        sums['sse'] += float(err.square().sum())
        sums['identity_sse'] += float(identity.square().sum())
        sums['input_sse'] += float(initial.square().sum())
        sums['count'] += err.numel()
    n = sums['count']
    if n < 1:
        raise ValueError('nonempty declared validation required')
    output, identity = (sums['sse']/n)**.5, (sums['identity_sse']/n)**.5
    return dict(output_log_RMSE=output, reference_change_log_RMS=identity,
        input_log_RMSE=(sums['input_sse']/n)**.5, selection_score=output+identity,
        observed_values=n, source_windows=len(items), validation_updates=0,
        scope='fixed native windows; NOT full-field quality acceptance', real_clean_truth=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('pretrain', 'adapt'), required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--random-ripple', action='store_true')
    args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.manual_seed(20261108)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl')
    manifest = read(args.root/'admission_manifest.json')
    records = manifest['records']
    profile = read(args.root/('random_ripple_profile.json' if args.random_ripple else 'aggregate_appearance_profile.json'))
    if args.random_ripple:
        # Explicit opt-in only; frozen old runs keep their original data path.
        from RandomRippleSonarData import example as ripple_example
        globals()['example'] = ripple_example
    out = args.root/args.phase
    if rank == 0:
        out.mkdir(exist_ok=False)
    dist.barrier()
    start = time.monotonic()
    model = NoFIRSonarMWCNN(args.width).to(device)
    warm = None
    if args.phase == 'adapt':
        previous = read(args.root/'pretrain/run_summary.json')
        warm = previous['selected']
        path = args.root/'pretrain'/warm['file']
        if sha(path) != warm['SHA256']:
            raise ValueError('selected public pretrained weight changed')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['model_family'] != model.family or checkpoint['feature_protocol'] != PROTOCOL:
            raise ValueError('explicit no-FIR warm family required')
        model.load_state_dict(checkpoint['model'], strict=True)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    lr = 1e-4 if args.phase == 'pretrain' else 3e-5
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=lr*.1)
    report = dict(status='initializing_actual_no_FIR_distributed_training', phase=args.phase,
        PID=os.getpid(), smoke=args.smoke, world_size=world, global_batch_samples=world,
        width=args.width, parameters=sum(p.numel() for p in model.parameters()),
        model_family=model.family, feature_protocol=PROTOCOL,
        steps_completed=0, steps_requested=args.steps, actual_global_examples=0,
        random_ripple_augmentation=args.random_ripple,
        noise_profile_SHA256=sha(args.root/('random_ripple_profile.json' if args.random_ripple else 'aggregate_appearance_profile.json')),
        input_frequency_FIR=False, output_frequency_FIR=False, actual_phase_only=True,
        warm_public_checkpoint=warm, phase_learning_rate=lr,
        source_manifest_SHA256=sha(args.root/'admission_manifest.json'),
        public_validation_and_native_holdouts_not_training=True,
        original_reference_NOT_clean_truth=True, real_quality_accepted=False,
        validation_history=[], checkpoints=[], rank_ledgers=[])
    def save_report():
        report['elapsed_s'] = time.monotonic()-start
        if rank == 0:
            save(out/'run_summary.json', report)
    save_report()
    checkpoints_at = {0, args.steps} | ({5000, 10000} if args.phase == 'pretrain' else {2000, 4000})
    if args.random_ripple:
        checkpoints_at |= set(range(4000 if args.phase == 'pretrain' else 2000, args.steps, 4000 if args.phase == 'pretrain' else 2000))
    checkpoints_at = {s for s in checkpoints_at if s <= args.steps}
    coverage = Counter()
    augmented_coverage = Counter()
    identity_coverage = Counter()
    source_domains = Counter()
    sources = Sources()
    ledger_path = out/f'actual_rank_{rank}_inputs.jsonl'
    try:
        def evaluate_and_save(step):
            dist.barrier()
            if rank == 0:
                report['status'] = 'validating_fixed_native_windows_without_optimizer_updates'
                save_report()
                metrics = validate(model, records, device, profile, args.phase, args.smoke)
                report['validation_history'].append(dict(step=step, metrics=metrics))
                file = out/f'checkpoint_{step:06d}.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), width=args.width, model_family=model.family,
                    feature_protocol=PROTOCOL, phase=args.phase, step=step,
                    admission_manifest_SHA256=report['source_manifest_SHA256'], no_FIR=True,
                    random_ripple_augmentation=args.random_ripple,
                    noise_profile_SHA256=report['noise_profile_SHA256']), file)
                report['checkpoints'].append(dict(step=step, file=file.name, SHA256=sha(file)))
                print(dict(stage='validation', phase=args.phase, step=step, metrics=metrics), flush=True)
                save_report()
            dist.barrier()
        evaluate_and_save(0)
        with ledger_path.open('x') as ledger:
            for step in range(1, args.steps+1):
                index = (step-1)*world+rank
                item = plan_source(records, args.phase, index)
                data, core, metadata = example(sources, item, index, args.phase, device, profile)
                features, mask, base, target = data
                ddp.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = ddp(features, mask, item['domain'])[core]
                loss = fidelity_loss(prediction, (target-base)[core], target[core], mask[core])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite actual no-FIR training loss')
                loss.backward()
                if step <= 2 and rank == 0:
                    report.setdefault('initial_update_gradient_checks', []).append(dict(step=step,
                        backbone_gradient_absolute_sum=float(model.network.head[0].weight.grad.abs().sum()),
                        output_head_gradient_absolute_sum=float(model.network.tail.weight.grad.abs().sum())))
                norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                coverage[item['id']] += 1
                if metadata['identity']:
                    identity_coverage[item['id']] += 1
                elif not args.random_ripple or metadata['noise']['actual_changed_values'] > 0:
                    augmented_coverage[item['id']] += 1
                source_domains[item['domain']] += 1
                metadata.update(step=step, rank=rank, loss=float(loss.detach()), gradient_norm=float(norm))
                ledger.write(json_text(metadata, sort_keys=True)+'\n')
                if step % 100 == 0 or step == args.steps:
                    ledger.flush()
                    reduced = torch.tensor([float(loss.detach())], device=device)
                    dist.all_reduce(reduced)
                    report.update(status='training_actual_no_FIR_multisource_updates', steps_completed=step,
                        actual_global_examples=step*world, mean_rank_loss=float(reduced/world),
                        learning_rate=optimizer.param_groups[0]['lr'])
                    save_report()
                    if rank == 0:
                        print(dict(stage='training', phase=args.phase, step=step,
                                   examples=step*world, loss=report['mean_rank_loss'],
                                   elapsed_s=report['elapsed_s']), flush=True)
                if step in checkpoints_at:
                    evaluate_and_save(step)
        save(out/f'rank_{rank}_receipt.json', dict(rank=rank, steps=args.steps, examples=sum(coverage.values()),
            actual_source_exposures=dict(coverage), actual_domain_exposures=dict(source_domains),
            actual_augmented_source_exposures=dict(augmented_coverage),
            actual_identity_source_exposures=dict(identity_coverage),
            ledger_SHA256=sha(ledger_path), peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device)))
        dist.barrier()
        if rank == 0:
            total = Counter()
            augmented_total = Counter()
            for r in range(world):
                receipt = read(out/f'rank_{r}_receipt.json')
                if receipt['steps'] != args.steps or receipt['examples'] != args.steps:
                    raise ValueError('every rank must perform each actual optimizer update')
                total.update(receipt['actual_source_exposures'])
                augmented_total.update(receipt['actual_augmented_source_exposures'])
                report['rank_ledgers'].append(dict(rank=r, receipt_SHA256=sha(out/f'rank_{r}_receipt.json')))
            roles = ('public_train',) if args.phase == 'pretrain' else ('native_train',)
            expected = {r['id'] for r in records if r['role'] in roles}
            if not args.smoke and not expected <= set(total):
                raise ValueError('declared training sources were not all actually used')
            if args.random_ripple and not args.smoke and not expected <= set(augmented_total):
                raise ValueError('every declared training source must actually receive random ripple noise')
            if any(r['role'].endswith('validation') and r['id'] in total for r in records):
                raise ValueError('held-out source entered optimization')
            best = min(report['validation_history'], key=lambda v: (v['metrics']['selection_score'], v['step']))
            report.update(status='completed_distributed_execution_smoke_NOT_quality' if args.smoke else
                'completed_actual_no_FIR_phase_training_NOT_quality_acceptance',
                selected=next(c for c in report['checkpoints'] if c['step'] == best['step']),
                actual_source_exposures=dict(total), actual_unique_sources=len(total),
                actual_augmented_source_exposures=dict(augmented_total),
                all_declared_training_sources_actually_augmented=expected <= set(augmented_total),
                all_declared_training_sources_seen=expected <= set(total),
                steps_completed=args.steps, actual_global_examples=args.steps*world)
            save_report()
        dist.barrier()
    except BaseException as exc:
        if rank == 0:
            report.update(status='failed_preserved_no_FIR_training_NO_automatic_restart', error=f'{type(exc).__name__}: {exc}')
            save_report()
        raise
    finally:
        del ddp, optimizer, model
        gc.collect()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
