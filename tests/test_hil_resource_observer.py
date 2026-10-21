import json

from standalone.observe_hil_host_resources import cgroup_memory, memory_info


def test_resource_reader_preserves_units_without_changing_files(tmp_path):
    source = tmp_path / "synthetic_meminfo"
    source.write_text("MemTotal: 1000 kB\nMemAvailable: 400 kB\nSwapTotal: 200 kB\nSwapFree: 1 kB\nOther: 12 kB\n")
    before = source.read_bytes()
    values = memory_info(source)
    assert values["MemAvailable_bytes"] == 400*1024
    assert values["SwapFree_bytes"] == 1024
    assert "Other_bytes" not in values
    assert source.read_bytes() == before


def test_cgroup_kernel_peak_events_and_unlimited_setting_are_explicit(tmp_path):
    for name, value in {"current": "123", "peak": "200", "max": "max", "swap.current": "0",
                        "events": "high 4\nmax 0\noom 0\noom_kill 0\n",
                        "stat": "anon 50\nfile 73\nshmem 0\nother 25\n"}.items():
        (tmp_path / ("memory."+name)).write_text(value)
    values = cgroup_memory(tmp_path)
    assert values["peak_bytes"] == 200
    assert values["max_bytes"] is None
    assert values["events"]["oom_kill"] == 0
    assert values["stat"] == dict(anon=50, file=73, shmem=0)
    json.dumps(values, allow_nan=False)


def test_already_gone_cgroup_does_not_invent_zero_measurements(tmp_path):
    assert cgroup_memory(tmp_path / "gone") == {}
