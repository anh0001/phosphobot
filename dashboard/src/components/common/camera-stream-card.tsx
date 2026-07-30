import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { CameraOff, PlugZap, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type React from "react";

// Default parameters for the streams (these can be passed as props or come from config)
const defaultQuality = 8;
const highQuality = 80;
let streamInstanceCounter = 0;

// Helper to compute stream URL based on current hostname, port, and query parameters
export const createStreamInstanceToken = () => {
  streamInstanceCounter += 1;
  return `${Date.now()}-${streamInstanceCounter}`;
};

export const getStreamUrl = (
  streamPath: string,
  quality: number,
  streamInstance: string,
  locationOverride: Pick<Location, "hostname" | "port"> = window.location,
) =>
  `http://${locationOverride.hostname}:${locationOverride.port}${streamPath}?quality=${quality}&streamInstance=${encodeURIComponent(streamInstance)}`;

export interface CameraStreamProps {
  id: number;
  title: string;
  streamPath: string;
  alt?: string;
  icon?: React.ReactNode;
  isRecording?: boolean;
  onRecordingToggle?: (id: number, isRecording: boolean) => void;
  showRecordingControls?: boolean;
  labelText?: string;
  /**
   * When false, the backend has released the underlying device (eg: /dev/video0)
   * so another process can use it. No frames are available until it is enabled.
   */
  isDeviceEnabled?: boolean;
  onDeviceToggle?: (id: number, isEnabled: boolean) => void;
  showDeviceControls?: boolean;
  isDeviceToggling?: boolean;
}

export const CardContentPiece = ({
  id,
  streamPath,
  alt,
  isRecording,
  showRecordingControls,
  quality,
  isDeviceEnabled = true,
}: {
  id: number;
  streamPath: string;
  alt?: string;
  isRecording: boolean;
  showRecordingControls: boolean;
  quality: number;
  isDeviceEnabled?: boolean;
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // Whether this card should actually show a live stream. A released device
  // has no frames to serve, so never try to connect to it.
  const isStreamActive =
    isDeviceEnabled && (isRecording || !showRecordingControls);

  const handleImageLoad = () => {
    setIsLoading(false);
    setHasError(false);
  };

  const handleImageError = () => {
    setIsLoading(false);
    setHasError(true);
  };

  // Connect / disconnect the MJPEG stream based on active state
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    if (isStreamActive) {
      setIsLoading(true);
      setHasError(false);
      const streamInstance = createStreamInstanceToken();
      const nextStreamUrl = getStreamUrl(streamPath, quality, streamInstance);

      img.src = "";
      img.removeAttribute("src");

      const timeoutId = window.setTimeout(() => {
        if (imgRef.current === img) {
          img.src = nextStreamUrl;
        }
      }, 0);

      return () => {
        window.clearTimeout(timeoutId);
        img.src = "";
      };
    } else {
      setIsLoading(false);
      setHasError(false);
      img.src = "";
    }

    return () => {
      if (img) {
        img.src = ""; // Disconnect when component unmounts
      }
    };
  }, [isStreamActive, streamPath, quality]);

  return (
    <div className="relative bg-muted">
      {hasError && isStreamActive && (
        <div className="flex items-center gap-1">
          <CameraOff className="size-6" />
          Stream Unavailable
        </div>
      )}
      {!isStreamActive && !isDeviceEnabled && (
        <div>
          <div className="flex items-center gap-1">
            <PlugZap className="size-6" />
            Camera Released
          </div>
          <p className="text-sm">
            The device is free for other processes. Turn the camera back on to
            view the feed.
          </p>
        </div>
      )}
      {!isStreamActive && isDeviceEnabled && (
        <div>
          <div className="flex items-center gap-1">
            <X className="size-6" />
            Feed Disabled
          </div>
          <p className="text-sm">Enable this camera to view the feed</p>
        </div>
      )}
      <img
        id={`view-video-${id}`}
        ref={imgRef}
        alt={alt}
        className={`w-full max-h-[360px] object-cover transition-opacity duration-300 ${
          isLoading || hasError || !isStreamActive ? "opacity-0" : "opacity-100"
        }`}
        onLoad={handleImageLoad}
        onError={handleImageError}
      />
    </div>
  );
};

export const CameraStreamCard = ({
  id,
  title,
  streamPath,
  alt = "Camera Stream",
  icon,
  isRecording = false,
  onRecordingToggle,
  showRecordingControls = false,
  labelText = "Record",
  isDeviceEnabled = true,
  onDeviceToggle,
  showDeviceControls = false,
  isDeviceToggling = false,
}: CameraStreamProps) => {
  const [quality, setQuality] = useState(defaultQuality);
  const toggleQuality = () => {
    setQuality(quality === defaultQuality ? highQuality : defaultQuality);
  };

  const handleRecordingChange = (checked: boolean) => {
    if (onRecordingToggle) {
      onRecordingToggle(id, checked);
    }
  };

  const handleDeviceChange = (checked: boolean) => {
    if (onDeviceToggle) {
      onDeviceToggle(id, checked);
    }
  };

  return (
    <Card className="overflow-hidden">
      {title !== "" && (
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl flex items-center gap-2">
            {icon}
            {title}
          </CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <CardContentPiece
          id={id}
          streamPath={streamPath}
          alt={alt}
          isRecording={isRecording}
          showRecordingControls={showRecordingControls}
          quality={quality}
          isDeviceEnabled={isDeviceEnabled}
        />
      </CardContent>
      <CardFooter className="justify-between">
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className="hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            onClick={toggleQuality}
          >
            Preview:{" "}
            {quality === defaultQuality ? "Low quality" : "High quality"}
          </Badge>
          {showDeviceControls && (
            <div className="flex items-center gap-2">
              <Switch
                id={`device-${id}`}
                checked={isDeviceEnabled}
                disabled={isDeviceToggling}
                onCheckedChange={handleDeviceChange}
                aria-label={
                  isDeviceEnabled
                    ? `Release camera ${id} for other processes`
                    : `Reconnect camera ${id}`
                }
              />
              <label
                htmlFor={`device-${id}`}
                className="text-sm font-medium leading-none"
              >
                {isDeviceEnabled ? "Camera on" : "Released"}
              </label>
            </div>
          )}
        </div>
        {showRecordingControls && (
          <div className="flex items-center gap-2">
            <Checkbox
              id={`record-${id}`}
              checked={isRecording}
              disabled={!isDeviceEnabled}
              onCheckedChange={handleRecordingChange}
            />
            <label
              htmlFor={`record-${id}`}
              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
            >
              {labelText}
            </label>
          </div>
        )}
      </CardFooter>
    </Card>
  );
};
