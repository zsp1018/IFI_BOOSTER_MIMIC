"""A concise numpy array implementation backed by SharedMemory + file locks.

Provides a lightweight `SyncedArray` class:
- Supports attaching by name across processes
- Read/write operations acquire a mutex (via file locking)
- Supports zero-copy view context management (holding the lock during the context)
- Supports create/attach semantics and resource cleanup

Design trade-offs:
- Use file-based flock for cross-process mutex for simplicity and portability
- Aims for simplicity and cross-platform behavior (Unix/Linux/macOS/Windows)
"""
from __future__ import annotations

import atexit
from multiprocessing import shared_memory
from typing import Tuple, Union
import fcntl
import os
import tempfile

import numpy as np


class SyncedArray:
    """A simple shared-memory backed numpy array for thread/process synchronization.

    Example usage:
        # Creator:
        arr = SyncedArray("myarr", shape=(10, 20), dtype="float32")
        arr.write(np.zeros((10, 20), dtype=np.float32))

        # Attach from another process:
        arr2 = SyncedArray.attach("myarr", shape=(10, 20), dtype="float32")
        data = arr2.read()

    Note: This implementation uses a file-based lock (flock) for cross-process
    synchronization.
    """

    def __init__(
        self,
        name: str,
        shape: Union[Tuple[int, ...], int],
        dtype: Union[str, np.dtype] = "float32",
        create: bool = True,
        owner_pid: int | None = None,
    ) -> None:
        self.name = name
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        # Convert dtype to np.dtype (accepts string or existing np.dtype)
        self.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype

        self._numel = int(np.prod(self.shape))
        self._size = int(self._numel * self.dtype.itemsize)

        # Decide owner pid: creator uses its own pid by default; attach uses parent pid
        if owner_pid is None:
            owner_pid = os.getpid() if create else os.getppid()
        self._owner_pid = int(owner_pid)

        # shared memory name: include owner pid to namespace lock and shm together
        self._shm_name = f"{self.name}_{self._owner_pid}_shm"
        try:
            if create:
                # create new shared memory block
                self.shm = shared_memory.SharedMemory(
                    name=self._shm_name, create=True, size=self._size
                )
            else:
                # attach existing
                self.shm = shared_memory.SharedMemory(name=self._shm_name)
        except FileExistsError:
            # If an object with the same name already exists and create=True,
            # try attaching to the existing shared memory instead
            if create:
                self.shm = shared_memory.SharedMemory(name=self._shm_name)
            else:
                raise

        # File lock used for cross-process mutex. All processes use the
        # same lock file path and call fcntl.flock.
        lock_dir = tempfile.gettempdir()
        # Include owner pid in lock file name to avoid collisions between
        # different main processes.
        self._lock_path = os.path.join(
            lock_dir, f"synced_array_{self.name}_{self._owner_pid}.lock"
        )
        # Open and retain the file descriptor (do not delete the file).
        # This descriptor is used with flock.
        self._lock_fd = open(self._lock_path, "a+")

        self._closed = False

        # Register exit cleanup (prevent resource leaks on unexpected exit)
        atexit.register(self.cleanup)

    # ---------------- basic operations ----------------
    def write(self, arr: np.ndarray) -> None:
        """Write the entire array (acquires mutex during the write)."""
        if self._closed:
            raise RuntimeError("SyncedArray is closed")
        data = np.asarray(arr, dtype=self.dtype)
        if data.size != self._numel:
            raise ValueError(
                f"Expected {self._numel} elements, got {data.size}"
            )
        if data.shape != self.shape:
            data = data.reshape(self.shape)

        # Use file exclusive lock to protect the write
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
            dst = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)  # type: ignore
            np.copyto(dst, data)
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def read(self) -> np.ndarray:
        """Read data (acquires mutex during read) and return a copy."""
        if self._closed:
            raise RuntimeError("SyncedArray is closed")
        # Acquire shared lock on read, allowing multiple concurrent readers
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_SH)
            src = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)  # type: ignore
            return src.copy()
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def modify_in_place(self, func):
        """Atomically modify a shared-memory view under an exclusive lock.

        `func` will receive a numpy view (not a copy) over the shared memory
        and may modify it in place. The lock is released after the call.
        """
        if self._closed:
            raise RuntimeError("SyncedArray is closed")
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
            dst = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)  # type: ignore
            # Call the user-provided function to modify dst in place
            func(dst)
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    # ---------------- lifecycle ----------------
    def cleanup(self) -> None:
        """Close and (if permitted) unlink the shared memory name.

        This operation is idempotent.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        except Exception:
            pass
        try:
            # Only attempt unlink in the creator scenario; otherwise it may
            # fail in other processes
            try:
                self.shm.unlink()
            except Exception:
                # Ignore unlink failures (not the creator or already removed)
                pass
        finally:
            # Close lock fd (do not delete the lock file itself)
            try:
                if hasattr(self, "_lock_fd") and self._lock_fd:
                    try:
                        self._lock_fd.close()
                    except Exception:
                        pass
            finally:
                self._lock_fd = None

    @staticmethod
    def attach(
        name: str,
        shape: Union[Tuple[int, ...], int],
        dtype: Union[str, np.dtype] = "float32",
    ) -> "SyncedArray":
        """Attach to an existing SyncedArray from another process.

        Try attaching using two candidate owner_pids: current process pid and
        parent pid, to be compatible with both in-process sequential attach
        and child-process attach scenarios. If neither succeeds, the original
        FileNotFoundError is raised.
        """
        # Try using the current pid
        last_exc: Exception | None = None
        for pid_candidate in (os.getpid(), os.getppid()):
            try:
                return SyncedArray(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    create=False,
                    owner_pid=pid_candidate,
                )
            except FileNotFoundError as e:
                last_exc = e
                # Try the next candidate
                continue
        # No success, raise the last exception
        if last_exc:
            raise last_exc
        # Should not reach here, but return a value for type-checkers
        return SyncedArray(name=name, shape=shape, dtype=dtype, create=False)


__all__ = ["SyncedArray"]
