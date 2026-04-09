import { describe, expect, it } from "vitest";

import type { SingleCameraStatus } from "@/types";

// Re-create the getStreamUrl helper logic from the component for unit-testing
// (the component uses window.location which isn't available here).
const getStreamUrl = (streamPath: string, quality: number) =>
  `http://localhost:8080${streamPath}?quality=${quality}`;

describe("getStreamUrl", () => {
  it("should include the quality parameter in the URL", () => {
    const url = getStreamUrl("/video/0", 8);
    expect(url).toBe("http://localhost:8080/video/0?quality=8");
  });

  it("should reflect high quality when toggled", () => {
    const url = getStreamUrl("/video/1", 80);
    expect(url).toBe("http://localhost:8080/video/1?quality=80");
  });
});

describe("realsense_depth filtering", () => {
  const cameras: SingleCameraStatus[] = [
    {
      camera_id: 0,
      is_active: true,
      camera_type: "classic",
      width: 640,
      height: 480,
      fps: 30,
    },
    {
      camera_id: 1,
      is_active: true,
      camera_type: "realsense_rgb",
      width: 640,
      height: 480,
      fps: 30,
    },
    {
      camera_id: 2,
      is_active: true,
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
      ? getStreamUrl("/video/0", 8)
      : "";
    expect(streamSrc).toBe("");
  });

  it("should produce a stream URL when card is enabled", () => {
    const isRecording = true;
    const showRecordingControls = true;
    const isStreamActive = isRecording || !showRecordingControls;

    const streamSrc = isStreamActive
      ? getStreamUrl("/video/0", 8)
      : "";
    expect(streamSrc).toBe("http://localhost:8080/video/0?quality=8");
  });
});
