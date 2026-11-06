"""Explicit new feature protocol, same original complex corruption bank.

The raw v7 input ledger remains reconstructible and is checked against the
completed v7 experiment. Only feature representation changes. Original MAT,
public complex phases, corruption geometry and q targets are never edited.
"""
from dataclasses import dataclass

from CarrierRelativeSonarFeatures import carrier_relative_features,PROTOCOL
from PhaseAwareSonarWavelet import PhaseAwareSonarWavelet
from PublicComplexPhaseExperiment import ComplexObservation,CompleteComplexBank,corrupt_complete,array_sha


class CarrierRelativeSonarWavelet(PhaseAwareSonarWavelet):
    family='native_carrier_relative_coherence_weighted_scalar_wavelet_consensus_v1'


@dataclass
class CarrierRelativeObservation(ComplexObservation):
    raw_features_sha256:str=''

    @classmethod
    def read(cls,row,path,device):
        ref=ComplexObservation.read(row,path,device)
        raw_digest=array_sha(ref.features)
        features,_=carrier_relative_features(ref.features,ref.support,complete_source=True)
        features.flags.writeable=False
        return cls(ref.sample_id,ref.path,ref.source_sha256,features,ref.logs,ref.support,ref.origin,raw_digest)


def relative_field(field,support):
    metadata=field.metadata
    metadata['raw_v7_noisy_features_sha256']=metadata['noisy_features_sha256']
    features,parameters=carrier_relative_features(field.features,support,complete_source=True)
    features.flags.writeable=False;field.features=features
    metadata.update(noisy_features_sha256=array_sha(features),phase_feature_protocol=PROTOCOL,
        phase_feature_statistics=parameters)
    return field


def corrupt_relative_complete(ref,profile,device,seed,scale):
    return relative_field(corrupt_complete(ref,profile,device,seed,scale),ref.support)


class CarrierRelativeComplexBank(CompleteComplexBank):
    def refresh(self,epoch):
        record=super().refresh(epoch)
        for ref,field in zip(self.references,self.fields):relative_field(field,ref.support)
        # super.refresh's ledger dictionaries refer to these same field
        # metadata, retaining both the original and transformed digests.
        record['phase_feature_protocol']=PROTOCOL
        return record

    def sample(self,rng,**options):
        features,noisy,target,support,metadata=super().sample(rng,**options)
        index=next(i for i,ref in enumerate(self.references) if ref.sample_id==metadata['sample_id'])
        raw=dict(metadata)
        raw['full_noisy_features_sha256']=(self.references[index].raw_features_sha256 if metadata['identity']
            else self.fields[index].metadata['raw_v7_noisy_features_sha256'])
        metadata['raw_v7_input_metadata']=raw
        metadata['phase_feature_protocol']=PROTOCOL
        return features,noisy,target,support,metadata
