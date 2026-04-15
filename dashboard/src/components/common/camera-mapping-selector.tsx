import { CameraStreamCard } from "@/components/common/camera-stream-card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useGlobalStore } from "@/lib/hooks";
import { fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { TriangleAlert } from "lucide-react";
import { useEffect } from "react";
import useSWR from "swr";

export interface CameraKeyMapperProps {
  modelKeys?: string[];
}

export default function CameraKeyMapper({ modelKeys }: CameraKeyMapperProps) {
  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
  });

  const cameraIds = serverStatus?.cameras.video_cameras_ids || [];
  const mapping = useGlobalStore((state) => state.cameraKeysMapping);
  const setMapping = useGlobalStore((state) => state.setCameraKeysMapping);

  useEffect(() => {
    if (modelKeys === undefined) {
      console.log("Model keys are undefined, clearing camera mapping");
      setMapping(null);
      return;
    }

    if (cameraIds.length > 0) {
      console.log("Setting initial camera mapping");
      const initial: Record<string, number> = {};
      modelKeys.forEach((key, idx) => {
        initial[key] =
          idx < cameraIds.length
            ? cameraIds[idx]
            : cameraIds[cameraIds.length - 1];
      });
      setMapping(initial);
    } else {
      console.log("No cameras available, clearing camera mapping");
      setMapping(null);
    }
  }, [JSON.stringify(cameraIds), JSON.stringify(modelKeys)]);

  const handleChange = (modelKey: string, camId: number) => {
    console.log("Updating mapping for", modelKey, "to camera ID", camId);
    const newValue = mapping
      ? { ...mapping, [modelKey]: camId }
      : { [modelKey]: camId };
    setMapping(newValue);
  };

  if (!serverStatus) {
    return <p>Loading camera list...</p>;
  }

  const hasModelKeys = modelKeys !== undefined && modelKeys.length > 0;

  return (
    <>
      <h2 className="text-lg font-semibold mb-4">Camera Mapping</h2>
      <p className="text-sm text-muted-foreground mb-4">
        Select the camera to be sent to each model key. The camera angles should
        match the dataset used for training the model.
      </p>
      <p className="text-xs text-muted-foreground mb-4">
        <code>main</code> should match your training main camera;{" "}
        <code>secondary_0</code> should match your training secondary camera.
      </p>
      {!hasModelKeys && (
        <>
          <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
            <TriangleAlert className="h-4 w-4" />
            Enter a valid model ID to map cameras to model inputs. Available
            camera feeds are shown below.
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            {cameraIds.map((cameraId) => (
              <CameraStreamCard
                key={cameraId}
                id={cameraId}
                title={`Camera ${cameraId}`}
                streamPath={`/video/${cameraId}`}
                alt={`Video Stream ${cameraId}`}
                showRecordingControls={false}
              />
            ))}
          </div>
        </>
      )}
      {hasModelKeys && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {mapping !== null &&
            modelKeys.map((key) => {
              const camId = mapping[key];
              return (
                <div key={key} className="flex flex-col gap-2">
                  <div>
                    <div className="text-lg font-medium">
                      <code>{key}</code>
                    </div>
                    <Select
                      value={camId?.toString() ?? ""}
                      onValueChange={(value) =>
                        handleChange(key, parseInt(value, 10))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select camera" />
                      </SelectTrigger>
                      <SelectContent>
                        {cameraIds.map((id) => (
                          <SelectItem key={id} value={id.toString()}>
                            Camera {id}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  {camId !== undefined && (
                    <CameraStreamCard
                      id={camId}
                      title={`Camera ${camId}`}
                      streamPath={`/video/${camId}`}
                      alt={`Video Stream ${camId}`}
                      showRecordingControls={false}
                    />
                  )}
                  {camId === undefined && (
                    <div className="flex items-center justify-center h-32 bg-muted rounded-md">
                      <p className="text-sm text-muted-foreground">
                        <TriangleAlert className="mr-2 h-4 w-4 text-red-500" />
                        No camera selected!
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      )}
    </>
  );
}
