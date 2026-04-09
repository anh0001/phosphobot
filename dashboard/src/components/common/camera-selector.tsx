"use client";

import { CardContentPiece } from "@/components/common/camera-stream-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetcher } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { Camera, Check, TriangleAlert } from "lucide-react";
import useSWR from "swr";

export interface CameraSelectorProps {
  onCameraSelect: (cameraId: number) => void;
  selectedCameraId: number;
  title?: string;
  description?: string;
}

export default function CameraSelector({
  onCameraSelect,
  selectedCameraId,
  title = "Select a camera for object detection",
  description = "The camera should have the same viewpoint as the image_key had during the training.",
}: CameraSelectorProps) {
  const {
    data: serverStatus,
    isLoading,
    error,
  } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
  });

  const cameraIds = serverStatus?.cameras.video_cameras_ids || [];

  const handleCameraSelect = (cameraId: number) => {
    onCameraSelect?.(cameraId);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <div className="flex items-center justify-center h-32">
          <p className="text-sm text-muted-foreground">
            Loading available cameras...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <Card className="border-destructive">
          <CardContent className="flex items-center justify-center h-32 pt-6">
            <div className="flex items-center text-destructive">
              <TriangleAlert className="mr-2 h-4 w-4" />
              <p className="text-sm">Failed to load camera information</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (cameraIds.length === 0) {
    return (
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <Card className="border-dashed">
          <CardContent className="flex items-center justify-center h-32 pt-6">
            <div className="flex items-center text-muted-foreground">
              <Camera className="mr-2 h-4 w-4" />
              <p className="text-sm">No cameras available</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <div className="ml-2">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {cameraIds.map((cameraId) => {
            const isSelected = selectedCameraId === cameraId;

            return (
              <Card
                key={cameraId}
                className={cn(
                  "cursor-pointer transition-all duration-200 hover:shadow-md",
                  isSelected
                    ? "ring-2 ring-primary border-primary shadow-md"
                    : "hover:border-primary/50",
                )}
                onClick={() => handleCameraSelect(cameraId)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center">
                      <Camera className="mr-2 h-4 w-4" />
                      Camera {cameraId}
                    </CardTitle>
                    {isSelected && (
                      <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary text-primary-foreground">
                        <Check className="h-3 w-3" />
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="rounded-md overflow-hidden bg-muted">
                    <CardContentPiece
                      id={cameraId}
                      streamPath={`/video/${cameraId}`}
                      alt={`Video Stream ${cameraId}`}
                      isRecording={false}
                      showRecordingControls={false}
                      quality={8}
                    />
                  </div>
                  <div className="mt-3 flex justify-center">
                    <Button
                      variant={isSelected ? "default" : "outline"}
                      size="sm"
                      className="w-full"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCameraSelect(cameraId);
                      }}
                    >
                      {isSelected ? (
                        <>
                          <Check className="mr-2 h-3 w-3" />
                          Selected
                        </>
                      ) : (
                        "Select Camera"
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
