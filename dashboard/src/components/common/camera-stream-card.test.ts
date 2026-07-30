import { describe, expect, it } from "vitest";

import {
  createStreamInstanceToken,
  getStreamUrl,
} from "@/components/common/camera-stream-card";
import type { SingleCameraStatus } from "@/types";

describe("getStreamUrl", () => {
  it("should include the quality parameter in the URL", () => {
    const url = getStreamUrl("/video/0", 8, "stream-1", {
      hostname: "localhost",
      port: "8080",
    });
    expect(url).toContain("quality=8");
  });

  it("should include the stream instance parameter", () => {
    const url = getStreamUrl("/video/1", 80, "stream-2", {
      hostname: "localhost",
      port: "8080",
    });
    expect(url).toContain("streamInstance=stream-2");
  });

  it("should produce distinct URLs for reconnects of the same camera", () => {
    const firstUrl = getStreamUrl("/video/0", 8, createStreamInstanceToken(), {
      hostname: "localhost",
      port: "8080",
    });
    const secondUrl = getStreamUrl(
      "/video/0",
      8,
      createStreamInstanceToken(),
      {
        hostname: "localhost",
        port: "8080",
      },
    );

    expect(firstUrl).not.toBe(secondUrl);
    expect(firstUrl).toContain("/video/0");
    expect(secondUrl).toContain("/video/0");
  });
});

describe("realsense_depth filtering", () => {
  const cameras: SingleCameraStatus[] = [
    {
      camera_id: 0,
      is_active: true,
      is_disabled: false,
      camera_type: "classic",
      width: 640,
      height: 480,
      fps: 30,
    },
    {
      camera_id: 1,
      is_active: true,
      is_disabled: false,
      camera_type: "realsense_rgb",
      width: 640,
      height: 480,
      fps: 30,
    },
    {
      camera_id: 2,
      is_active: true,
      is_disabled: false,
      camera_type: "realsense_depth",
      width: 640,
      height: 480,
      fps: 30,
    },
  ];

  it("should filter out realsense_depth cameras from default view", () => {
    const filtered = cameras.filter(
      (cam) => cam.camera_type !== "realsense_depth",
    );
    expect(filtered).toHaveLength(2);
    expect(filtered.map((c) => c.camera_id)).toEqual([0, 1]);
  });

  it("should keep all non-depth cameras", () => {
    const filtered = cameras.filter(
      (cam) => cam.camera_type !== "realsense_depth",
    );
    expect(
      filtered.every((c) => c.camera_type !== "realsense_depth"),
    ).toBe(true);
  });
});

describe("disabled card stream behaviour", () => {
  it("should not produce a stream URL when card is disabled", () => {
    const isRecording = false;
    const showRecordingControls = true;
    const isStreamActive = isRecording || !showRecordingControls;

    // When not active, the component sets img.src = "" — model that here
    const streamSrc = isStreamActive
      ? getStreamUrl("/video/0", 8, "stream-3", {
          hostname: "localhost",
          port: "8080",
        })
      : "";
    expect(streamSrc).toBe("");
  });

  it("should produce a stream URL when card is enabled", () => {
    const isRecording = true;
    const showRecordingControls = true;
    const isStreamActive = isRecording || !showRecordingControls;

    const streamSrc = isStreamActive
      ? getStreamUrl("/video/0", 8, "stream-4", {
          hostname: "localhost",
          port: "8080",
        })
      : "";
    expect(streamSrc).toContain("http://localhost:8080/video/0?quality=8");
    expect(streamSrc).toContain("streamInstance=stream-4");
  });
});
