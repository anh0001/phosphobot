import { AddZMQCameraModal } from "@/components/common/add-zmq-camera-modal";
import { CameraStreamCard } from "@/components/common/camera-stream-card";
import { Button } from "@/components/ui/button";
import { useCameraControls } from "@/lib/hooks";
import { cn, fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AdminSettings, ServerStatus } from "@/types";
import { PlugZap, RotateCw, Video } from "lucide-react";
import { useCallback, useState } from "react";
import useSWR from "swr";

export function ViewVideoPage({ labelText }: { labelText?: string }) {
  if (!labelText) labelText = "Camera Stream";
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isZMQModalOpen, setIsZMQModalOpen] = useState(false);
  const [togglingCameraIds, setTogglingCameraIds] = useState<number[]>([]);
  const [isTogglingAll, setIsTogglingAll] = useState(false);

  const { data: serverStatus, mutate: mutateStatus } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const { data: adminSettings, mutate: mutateSettings } = useSWR<AdminSettings>(
    "/admin/settings",
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const { updateCameraRecording, isCameraEnabled } = useCameraControls(
    adminSettings,
    mutateSettings,
  );

  const camerasStatus = serverStatus?.cameras.cameras_status ?? [];
  const hasReleasedCamera = camerasStatus.some((cam) => cam.is_disabled);

  // Releasing a device frees it (eg: /dev/video0) for another process.
  const toggleCameraDevice = useCallback(
    async (cameraId: number, isEnabled: boolean) => {
      setTogglingCameraIds((ids) => [...ids, cameraId]);
      try {
        await fetchWithBaseUrl(
          `/cameras/${cameraId}/${isEnabled ? "enable" : "disable"}`,
          "POST",
        );
        await mutateStatus();
      } finally {
        setTogglingCameraIds((ids) => ids.filter((id) => id !== cameraId));
      }
    },
    [mutateStatus],
  );

  const toggleAllCameraDevices = useCallback(
    async (isEnabled: boolean) => {
      setIsTogglingAll(true);
      try {
        await fetchWithBaseUrl(
          `/cameras/${isEnabled ? "enable-all" : "disable-all"}`,
          "POST",
        );
        await mutateStatus();
      } finally {
        setIsTogglingAll(false);
      }
    },
    [mutateStatus],
  );

  return (
    <>
      <div className="mb-2 flex justify-end gap-x-2">
        <Button
          variant="outline"
          onClick={() => toggleAllCameraDevices(hasReleasedCamera)}
          disabled={isTogglingAll || isRefreshing || camerasStatus.length === 0}
          title={
            hasReleasedCamera
              ? "Re-open every camera device"
              : "Release every camera device so other processes can use them"
          }
        >
          <div className="flex items-center gap-2">
            <PlugZap className="h-4 w-4" />
            {hasReleasedCamera
              ? "Reconnect all cameras"
              : "Release all cameras"}
          </div>
        </Button>
        <Button
          variant="outline"
          className="ml-2"
          onClick={() => setIsZMQModalOpen(true)}
        >
          Add ZMQ Camera
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            setIsRefreshing(true);
            fetchWithBaseUrl("/cameras/refresh", "POST").then(() => {
              mutateStatus();
              mutateSettings();
              setIsRefreshing(false);
            });
          }}
          disabled={isRefreshing}
        >
          <div className="flex items-center gap-2">
            <RotateCw
              className={cn("h-4 w-4", isRefreshing && "animate-spin")}
            />
            Rescan cameras...
          </div>
        </Button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {isRefreshing && (
          <div className="col-span-1 sm:col-span-2 md:col-span-3 text-center text-muted-foreground">
            <p>
              Disconnecting the camera streams and restarting camera
              discovery...
            </p>
          </div>
        )}
        {!isRefreshing &&
          camerasStatus
            .filter((cam) => cam.camera_type !== "realsense_depth")
            .map((cam) => {
              return (
                <CameraStreamCard
                  key={cam.camera_id}
                  id={cam.camera_id}
                  title={`Camera ${cam.camera_id}`}
                  streamPath={`/video/${cam.camera_id}`}
                  alt={`Video Stream ${cam.camera_id}`}
                  icon={<Video className="h-4 w-4" />}
                  isRecording={isCameraEnabled(cam.camera_id)}
                  onRecordingToggle={updateCameraRecording}
                  showRecordingControls={true}
                  labelText={labelText}
                  isDeviceEnabled={!cam.is_disabled}
                  onDeviceToggle={toggleCameraDevice}
                  showDeviceControls={true}
                  isDeviceToggling={
                    isTogglingAll || togglingCameraIds.includes(cam.camera_id)
                  }
                />
              );
            })}
      </div>
      <AddZMQCameraModal
        open={isZMQModalOpen}
        onOpenChange={setIsZMQModalOpen}
      />
    </>
  );
}
