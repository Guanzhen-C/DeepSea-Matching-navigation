"""ROS 2 PointCloud2 adapter for the OceanSim multibeam echosounder."""

from __future__ import annotations

import numpy as np
import omni.graph.core as og

from isaacsim.oceansim.sensors.MultibeamEchosounderSensor import (
    MultibeamEchosounderSensor,
)


class MultibeamEchosounderSensor_ROS(MultibeamEchosounderSensor):
    """Publish valid bathymetric returns through ROS2PublishPointCloud.

    The complete per-beam ping remains available from :meth:`scan` and through
    ``ping.as_sonar_ranges_dict()``.  The ROS bridge publishes XYZ points using
    ``sensor_msgs/PointCloud2`` without requiring an extra message package.
    """

    def __init__(
        self,
        *args,
        og_node=None,
        topic_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._og_node = og_node
        self._topic_name = topic_name

    def initialize(self, physics_sim_view=None, og_node=None) -> None:
        if og_node is not None:
            self._og_node = og_node
        super().initialize(physics_sim_view)
        if self._og_node is not None:
            og.Controller.attribute(
                self._og_node.get_attribute("inputs:frameId")
            ).set(self._frame_id)
            if self._topic_name is not None:
                og.Controller.attribute(
                    self._og_node.get_attribute("inputs:topicName")
                ).set(self._topic_name)

    def _publish_ping(self, ping) -> None:
        if self._og_node is None:
            return
        points = np.ascontiguousarray(
            ping.valid_points(frame="sensor"), dtype=np.float32
        )
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:timeStamp")
        ).set(float(ping.timestamp))
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:bufferSize")
        ).set(points.nbytes)
        og.Controller.attribute(self._og_node.get_attribute("inputs:data")).set(
            points
        )

    def scan(self, timestamp: float | None = None):
        ping = super().scan(timestamp=timestamp)
        self._publish_ping(ping)
        return ping
