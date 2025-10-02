import serial


class SwitchBoard:
    """USB 100:1 switch with ACK sync."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 2.0):
        if serial is None:
            raise RuntimeError("pyserial not available")
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)

    def route(self, idx: int):
        if not 1 <= idx <= 100:
            raise ValueError("Pixel index must be 1-100")
        self.ser.write(f"{idx}\n".encode())
        resp = self.ser.readline()
        if b"ACK" not in resp:
            raise TimeoutError(
                f"Switch did not ACK for pixel {idx}. Response: {resp.decode(errors='ignore')}"
            )

    def close(self):
        self.ser.close()


class DummySwitchBoard:
    def route(self, *_):
        pass

    def close(self):
        pass
