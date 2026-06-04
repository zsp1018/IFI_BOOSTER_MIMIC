from typing import Optional
import evdev
import threading
from dataclasses import dataclass
import time
import termios
import tty
import select
import atexit
import sys


@dataclass
class JoystickConfig:
    max_vx: float = 1.
    max_vy: float = 1.
    max_vyaw: float = 1.
    control_threshold: float = 0.1
    # logitech
    custom_mode_button: evdev.ecodes = evdev.ecodes.BTN_A
    rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_B
    x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    yaw_axis: evdev.ecodes = evdev.ecodes.ABS_RX

    # xiaoji
    # custom_mode_button: evdev.ecodes = evdev.ecodes.BTN_B
    # rl_gait_button: evdev.ecodes = evdev.ecodes.BTN_A
    # x_axis: evdev.ecodes = evdev.ecodes.ABS_Y
    # y_axis: evdev.ecodes = evdev.ecodes.ABS_X
    # yaw_axis: evdev.ecodes = evdev.ecodes.ABS_RX


class RemoteControlService:
    """Service for handling joystick remote control input without display dependencies."""

    def __init__(self, config: Optional[JoystickConfig] = None):
        """Initialize remote control service with optional configuration."""
        self.config = config or JoystickConfig()
        self._lock = threading.Lock()
        self._running = True
        try:
            self._init_joystick()
            self._start_joystick_thread()
        except Exception as e:
            print(f"{e}, downgrade to keyboard control")
            self._init_keyboard_control()
            self._start_keyboard_thread()

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

    def get_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return "Joystick left axis for forward/backward/left/right, right axis for rotation left/right"
        return "Press keyboard 'w'/'s' to increase/decrease vx; Press 'a'/'d' to increase/decrease vy; Press 'q'/'e' to increase/decrease vyaw, press 'Space' to stop."

    def get_custom_mode_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return "Press joystick button X to start custom mode."
        return "Press keyboard 'x' to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return "Press joystick button A to start rl Gait."
        return "Press keyboard 'r' to start rl Gait."

    def _init_keyboard_control(self):
        self.joystick = None
        self.joystick_runner = None
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False

    def _start_keyboard_thread(self):
        # Start a thread that reads stdin in cbreak mode and dispatches presses.
        self.keyboard_runner = threading.Thread(target=self._keyboard_listener, daemon=True)
        # Save original terminal attrs so we can restore later
        try:
            if sys.stdin.isatty():
                self._stdin_tty = True
                self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            else:
                self._stdin_tty = False
                self._old_termios = None
        except Exception:
            self._stdin_tty = False
            self._old_termios = None

        # Ensure we attempt to clean up terminal on process exit
        try:
            atexit.register(self.close)
        except Exception:
            pass

        self.keyboard_runner.start()

    def _keyboard_listener(self):
        # Use cbreak mode so we can read key presses without requiring Enter.
        fd = None
        try:
            if not getattr(self, "_stdin_tty", False):
                # stdin is not a tty; nothing to do
                return
            fd = sys.stdin.fileno()
            tty.setcbreak(fd)
            while self._running:
                # small timeout to allow clean shutdown
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":  # Ctrl-C
                        # Let the main program handle KeyboardInterrupt
                        continue
                    if ch == " ":
                        key = "space"
                    else:
                        key = ch
                    try:
                        self._handle_keyboard_press(key)
                    except Exception:
                        # swallow handler errors to keep listener alive
                        pass
        finally:
            # restore terminal settings if we changed them
            try:
                if fd is not None and getattr(self, "_old_termios", None) is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _handle_keyboard_press(self, key):
        if key == "x":
            self.keyboard_start_custom_mode = True
        if key == "r":
            self.keyboard_start_rl_gait = True
        if key == "w":
            old_x = self.vx
            self.vx += 0.1
            self.vx = min(self.vx, self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "s":
            old_x = self.vx
            self.vx -= 0.1
            self.vx = max(self.vx, -self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "a":
            old_y = self.vy
            self.vy += 0.1
            self.vy = min(self.vy, self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "d":
            old_y = self.vy
            self.vy -= 0.1
            self.vy = max(self.vy, -self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "q":
            old_yaw = self.vyaw
            self.vyaw += 0.1
            self.vyaw = min(self.vyaw, self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "e":
            old_yaw = self.vyaw
            self.vyaw -= 0.1
            self.vyaw = max(self.vyaw, -self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "space":
            self.vx = 0
            self.vy = 0
            self.vyaw = 0
            print("FULL STOP")

    def _init_joystick(self) -> None:
        """Initialize and validate joystick connection using evdev."""
        try:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            joystick = None

            for device in devices:
                caps = device.capabilities()
                # print(f"Device {device.name}:")
                # print(f"Capabilities: {device.capabilities(verbose=True)}")

                # Check for both absolute axes and keys
                if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                    abs_info = caps.get(evdev.ecodes.EV_ABS, [])
                    # Look for typical gamepad axes
                    axes = [code for (code, info) in abs_info]
                    if all(code in axes for code in [self.config.x_axis, self.config.y_axis, self.config.yaw_axis]):
                        absinfo = {}
                        for code, info in abs_info:
                            absinfo[code] = info
                        self.axis_ranges = {
                            self.config.x_axis: absinfo[self.config.x_axis],
                            self.config.y_axis: absinfo[self.config.y_axis],
                            self.config.yaw_axis: absinfo[self.config.yaw_axis],
                        }
                        print(f"Found suitable joystick: {device.name}")
                        joystick = device
                        break

            if not joystick:
                raise RuntimeError("No suitable joystick found")

            self.joystick = joystick
            print(f"Selected joystick: {joystick.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}")

    def _start_joystick_thread(self):
        """Start joystick polling thread."""
        self.joystick_runner = threading.Thread(target=self._run_joystick)
        self.joystick_runner.daemon = True
        self.joystick_runner.start()

    def start_custom_mode(self) -> bool:
        """Check if custom mode button is pressed."""
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return self.joystick.active_keys() == [self.config.custom_mode_button]
        return self.keyboard_start_custom_mode

    def start_rl_gait(self) -> bool:
        """Check if gait button is pressed."""
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            return self.joystick.active_keys() == [self.config.rl_gait_button]
        return self.keyboard_start_rl_gait

    def _run_joystick(self):
        """Poll joystick events."""
        while self._running:
            try:
                # read one event
                event = self.joystick.read_one()
                if event:
                    if event.type == evdev.ecodes.EV_ABS:
                        self._handle_axis(event.code, event.value)
                else:
                    time.sleep(0.01)
            except Exception as e:
                if not self._running:  # If the exception was caused by shutdown, no need to log
                    break
                print(f"Error in joystick polling loop: {e}")
                time.sleep(0.05)

    def _handle_axis(self, code: int, value: int):
        try:
            """Handle axis events."""
            if code == self.config.x_axis:
                self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)
                # print("value x:", self.vx)
            elif code == self.config.y_axis:
                self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)
                # print("value y:", self.vy)
            elif code == self.config.yaw_axis:
                self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)
                # print("value yaw:", self.vyaw)
        except Exception:
            raise

    def _scale(self, value: float, max: float, threshold: float, axis_code: int) -> float:
        """Scale joystick input to velocity command using actual axis ranges."""
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max

        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max
        # print(f"Axis {axis_code}, value {value} min_in {min_in}, max_in {max_in}: {value} => {mapped_value}")

        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def get_vx_cmd(self) -> float:
        """Get forward velocity command."""
        with self._lock:
            return self.vx

    def get_vy_cmd(self) -> float:
        """Get lateral velocity command."""
        with self._lock:
            return self.vy

    def get_vyaw_cmd(self) -> float:
        """Get yaw velocity command."""
        with self._lock:
            return self.vyaw

    def close(self):
        """Clean up resources."""
        self._running = False
        # try restore stdin terminal settings if we changed them
        try:
            if getattr(self, "_stdin_tty", False) and getattr(self, "_old_termios", None) is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        if hasattr(self, "joystick") and getattr(self, "joystick") is not None:
            try:
                self.joystick.close()
            except Exception as e:
                print(f"Error closing joystick: {e}")
        if hasattr(self, "joystick_runner") and getattr(self, "joystick_runner") is not None:
            try:
                self.joystick_runner.join(timeout=1.0)
                if self.joystick_runner.is_alive():
                    print("Joystick thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for joystick thread to end: {e}")
        if hasattr(self, "keyboard_runner") and getattr(self, "keyboard_runner") is not None:
            try:
                self.keyboard_runner.join(timeout=1.0)
                if self.keyboard_runner.is_alive():
                    print("Keyboard thread didn't exit within the time limit")
            except Exception as e:
                print(f"Error waiting for keyboard thread to end: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
