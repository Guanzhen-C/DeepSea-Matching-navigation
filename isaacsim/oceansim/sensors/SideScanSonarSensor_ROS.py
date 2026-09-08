"""ROS 2 mono8 image adapter for the OceanSim side-scan sonar."""

from __future__ import annotations

import numpy as np
import omni.graph.core as og

from isaacsim.oceansim.sensors.SideScanSonarSensor import SideScanSonarSensor


class SideScanSonarSensor_ROS(SideScanSonarSensor):
    """Publish the accumulated side-scan waterfall as ``sensor_msgs/Image``."""

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
            og.Controller.attribute(
                self._og_node.get_attribute("inputs:encoding")
            ).set("mono8")

    def _publish_ping(self, ping) -> None:
        if self._og_node is None:
            return
        image = np.ascontiguousarray(self.get_waterfall(), dtype=np.uint8)
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:timeStamp")
        ).set(float(ping.timestamp))
        og.Controller.attribute(self._og_node.get_attribute("inputs:width")).set(
            int(image.shape[1])
        )
        og.Controller.attribute(self._og_node.get_attribute("inputs:height")).set(
            int(image.shape[0])
        )
        og.Controller.attribute(
            self._og_node.get_attribute("inputs:bufferSize")
        ).set(image.nbytes)
        og.Controller.attribute(self._og_node.get_attribute("inputs:data")).set(
            image.reshape(-1)
        )

    def scan(self, timestamp: float | None = None):
        ping = super().scan(timestamp=timestamp)
        self._publish_ping(ping)
        return ping
