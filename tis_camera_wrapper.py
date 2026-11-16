# -*- coding: utf-8 -*-
"""
TIS Camera Wrapper Module
Provides OpenCV-compatible interface for The Imaging Source (TIS) cameras

This module wraps the TIS_API to provide a cv2.VideoCapture-like interface,
allowing seamless integration with existing OpenCV-based video processing pipelines.

Supported cameras: DFK 33UX178, and other TIS cameras using tisgrabber DLL

Usage:
    from tis_camera_wrapper import TISCameraWrapper
    
    cap = TISCameraWrapper(
        dll_path='./tisgrabber_x64.dll',
        device_name='DFK 33UX178'
    )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
"""

import time
import numpy as np


class TISCameraWrapper:
    """
    OpenCV-compatible wrapper for TIS (The Imaging Source) cameras.
    
    Provides the same interface as cv2.VideoCapture for seamless integration
    with existing video processing code.
    
    Attributes:
        dll_path (str): Path to tisgrabber DLL
        device_name (str): Substring to match camera name
        ic: IC Imaging Control handle
        grabbers: List of grabber handles
        device_names: List of available device names
        camera_idx: Index of selected camera
        _is_opened (bool): Camera connection status
        _frame_width (int): Frame width in pixels
        _frame_height (int): Frame height in pixels
    """
    
    def __init__(self, dll_path='./tisgrabber_x64.dll', device_name='DFK 33UX178', 
                 camera_index=None, logger=None):
        """
        Initialize TIS camera wrapper.
        
        Args:
            dll_path: Path to tisgrabber_x64.dll
            device_name: Substring to match in device unique name (e.g., 'DFK 33UX178')
            camera_index: Optional specific camera index (overrides device_name search)
            logger: Optional logger instance for error logging
        """
        self.dll_path = dll_path
        self.device_name = device_name
        self.requested_camera_index = camera_index
        self.logger = logger
        
        # TIS API handles
        self.ic = None
        self.grabbers = None
        self.device_names = None
        self.camera_idx = None
        
        # State tracking
        self._is_opened = False
        self._is_capturing = False
        self._frame_width = 0
        self._frame_height = 0
        self._consecutive_failures = 0
        self._max_failures = 10
        
        # Performance tracking
        self._last_frame_time = 0
        self._frame_count = 0
        
        # Initialize camera
        self._initialize()
    
    def _initialize(self):
        """Initialize TIS camera using TIS_API."""
        try:
            # Import TIS_API - must be in the same directory or Python path
            from TIS_API import IC_GetImage, IC_Initial, IC_Release, IC_StartCapture
            
            # Store API functions for later use
            self._IC_GetImage = IC_GetImage
            self._IC_Release = IC_Release
            self._IC_StartCapture = IC_StartCapture
            
            # Initialize IC library
            print(f"[TIS] Initializing with DLL: {self.dll_path}")
            self.ic, self.grabbers, self.device_names = IC_Initial(self.dll_path)
            
            if not self.device_names:
                error_msg = "[TIS] No TIS cameras found"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("tis_camera_not_found", error_msg)
                return
            
            # Find camera index
            if self.requested_camera_index is not None:
                if self.requested_camera_index < len(self.device_names):
                    self.camera_idx = self.requested_camera_index
                else:
                    error_msg = f"[TIS] Invalid camera index {self.requested_camera_index}, only {len(self.device_names)} cameras available"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("tis_invalid_index", error_msg)
                    return
            else:
                self.camera_idx = self._find_camera_index(self.device_names, self.device_name)
            
            if self.camera_idx is None:
                return
            
            print(f"[TIS] Using camera [{self.camera_idx}]: {self.device_names[self.camera_idx]}")
            
            # Start capture
            IC_StartCapture(self.ic, self.grabbers)
            self._is_capturing = True
            self._is_opened = True
            
            # Get first frame to determine dimensions
            self._detect_frame_dimensions()
            
            print(f"[TIS] Camera initialized: {self._frame_width}x{self._frame_height}")
            if self.logger:
                self.logger.log_system_event("tis_camera_initialized", {
                    "device": self.device_names[self.camera_idx],
                    "resolution": f"{self._frame_width}x{self._frame_height}"
                })
            
        except ImportError as e:
            error_msg = f"[TIS] Failed to import TIS_API: {e}"
            print(error_msg)
            print("[TIS] Make sure TIS_API.py and tisgrabber_x64.dll are in the correct location")
            if self.logger:
                self.logger.log_error("tis_import_error", error_msg)
        except Exception as e:
            error_msg = f"[TIS] Initialization error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("tis_init_error", error_msg)
    
    def _find_camera_index(self, device_names, name_substring):
        """
        Find camera index by matching device name substring.
        
        Args:
            device_names: List of available device names
            name_substring: Substring to search for in device names
            
        Returns:
            Camera index if found, None otherwise
        """
        if not device_names:
            error_msg = "[TIS] No TIS cameras found"
            print(error_msg)
            if self.logger:
                self.logger.log_error("tis_no_cameras", error_msg)
            return None
        
        if not name_substring:
            return 0
        
        needle = name_substring.lower()
        for index, device_name in enumerate(device_names):
            if needle in device_name.lower():
                return index
        
        # Not found - list available cameras
        available = "\n".join(f"  [{i}] {n}" for i, n in enumerate(device_names))
        error_msg = f'[TIS] Cannot find camera containing "{name_substring}". Available cameras:\n{available}'
        print(error_msg)
        if self.logger:
            self.logger.log_error("tis_camera_not_found", error_msg, {"available": device_names})
        return None
    
    def _detect_frame_dimensions(self):
        """Detect frame dimensions from first captured frame."""
        if not self._is_opened:
            return
        
        # Try to get a frame to detect dimensions
        max_attempts = 30  # Wait up to 3 seconds
        for _ in range(max_attempts):
            frame = self._IC_GetImage(self.ic, self.grabbers, self.camera_idx)
            if not isinstance(frame, int) or frame != 0:
                if isinstance(frame, np.ndarray):
                    self._frame_height, self._frame_width = frame.shape[:2]
                    return
            time.sleep(0.1)
        
        print("[TIS] Warning: Could not detect frame dimensions")
    
    def isOpened(self):
        """
        Check if camera is opened and ready.
        
        Returns:
            bool: True if camera is ready for capture
        """
        return self._is_opened and self._is_capturing
    
    def read(self):
        """
        Read a frame from the camera.
        
        Returns:
            tuple: (success: bool, frame: np.ndarray or None)
                - success: True if frame was read successfully
                - frame: BGR image as numpy array, or None if failed
        """
        if not self._is_opened:
            return False, None
        
        try:
            frame = self._IC_GetImage(self.ic, self.grabbers, self.camera_idx)
            
            # Check for invalid frame (TIS returns 0 on failure)
            if isinstance(frame, int) and frame == 0:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    error_msg = f"[TIS] Too many consecutive failures ({self._consecutive_failures})"
                    print(error_msg)
                    if self.logger:
                        self.logger.log_error("tis_capture_failures", error_msg)
                return False, None
            
            # Valid frame received
            self._consecutive_failures = 0
            self._frame_count += 1
            self._last_frame_time = time.time()
            
            # IMPORTANT: Copy frame to avoid buffer overwrite issues
            # TIS API may reuse internal buffer
            frame = frame.copy()
            
            return True, frame
            
        except Exception as e:
            error_msg = f"[TIS] Frame read error: {e}"
            print(error_msg)
            if self.logger:
                self.logger.log_error("tis_read_error", error_msg)
            return False, None
    
    def get(self, prop_id):
        """
        Get camera property value (OpenCV compatibility).
        
        Args:
            prop_id: OpenCV property ID (cv2.CAP_PROP_*)
            
        Returns:
            Property value or 0 if not supported
        """
        import cv2
        
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frame_width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frame_height
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0  # Default estimate for TIS cameras
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        else:
            return 0
    
    def set(self, prop_id, value):
        """
        Set camera property (OpenCV compatibility).
        
        Note: Most properties are not supported for TIS cameras through this wrapper.
        Use TIS-specific APIs for advanced camera control.
        
        Args:
            prop_id: OpenCV property ID
            value: Property value
            
        Returns:
            bool: True if property was set successfully
        """
        # TIS camera properties would need to be set through IC_SetPropertyValue
        # This is a simplified implementation for compatibility
        print(f"[TIS] Warning: Property {prop_id} setting not implemented")
        return False
    
    def release(self):
        """Release camera resources."""
        if self._is_opened:
            try:
                print("[TIS] Releasing camera resources...")
                self._IC_Release(self.ic, self.grabbers)
                self._is_opened = False
                self._is_capturing = False
                print("[TIS] Camera released")
                if self.logger:
                    self.logger.log_system_event("tis_camera_released", {
                        "total_frames": self._frame_count
                    })
            except Exception as e:
                error_msg = f"[TIS] Release error: {e}"
                print(error_msg)
                if self.logger:
                    self.logger.log_error("tis_release_error", error_msg)
    
    def get_device_info(self):
        """
        Get information about the connected device.
        
        Returns:
            dict: Device information including name, index, resolution
        """
        return {
            "device_name": self.device_names[self.camera_idx] if self.device_names and self.camera_idx is not None else "Unknown",
            "camera_index": self.camera_idx,
            "resolution": f"{self._frame_width}x{self._frame_height}",
            "is_opened": self._is_opened,
            "is_capturing": self._is_capturing,
            "total_frames": self._frame_count,
            "available_devices": self.device_names or []
        }
    
    def list_available_cameras(self):
        """
        List all available TIS cameras.
        
        Returns:
            list: List of available device names with indices
        """
        if not self.device_names:
            return []
        return [f"[{i}] {name}" for i, name in enumerate(self.device_names)]
    
    def __del__(self):
        """Destructor - ensure resources are released."""
        self.release()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release resources."""
        self.release()
        return False


# Utility function for quick camera testing
def test_tis_camera(dll_path='./tisgrabber_x64.dll', device_name='DFK 33UX178'):
    """
    Quick test function to verify TIS camera connection.
    
    Args:
        dll_path: Path to tisgrabber DLL
        device_name: Camera name to search for
    """
    import cv2
    
    print("=" * 50)
    print("TIS Camera Test")
    print("=" * 50)
    
    cap = TISCameraWrapper(dll_path=dll_path, device_name=device_name)
    
    if not cap.isOpened():
        print("Failed to open camera!")
        return False
    
    print("\nDevice Info:")
    info = cap.get_device_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nStarting video stream (press 'q' to quit)...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS text to frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('TIS Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\nTest completed: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TIS Camera Wrapper")
    parser.add_argument("--dll", default="./tisgrabber_x64.dll", help="Path to tisgrabber DLL")
    parser.add_argument("--device", default="DFK 33UX178", help="Device name to search for")
    args = parser.parse_args()
    
    test_tis_camera(args.dll, args.device)
