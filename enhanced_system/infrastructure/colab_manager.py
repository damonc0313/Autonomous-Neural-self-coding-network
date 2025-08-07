"""
Google Colab Infrastructure Manager

Handles Colab-specific optimizations including timeout management, 
Google Drive integration, session recovery, and resource monitoring.
"""

import os
import time
import gzip
import pickle
import shutil
import gc
import threading
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import psutil
import subprocess
from contextlib import contextmanager

import torch
import numpy as np
from IPython.display import display, HTML, Javascript
import ipywidgets as widgets

from ..core.protocols import (
    CheckpointManager, TrainingState, ResourceManager, 
    SystemConfig, EvolutionMetrics
)


class ColabCheckpointManager:
    """Production-grade checkpoint manager optimized for Google Colab."""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive/neuroevolution", 
                 max_versions: int = 5, compression_level: int = 6):
        """
        Initialize checkpoint manager.
        
        Args:
            drive_path: Google Drive path for checkpoints
            max_versions: Maximum checkpoint versions to keep
            compression_level: Gzip compression level (1-9)
        """
        self.drive_path = Path(drive_path)
        self.max_versions = max_versions
        self.compression_level = compression_level
        self.metadata_file = self.drive_path / "checkpoint_metadata.json"
        
        # Ensure directories exist
        self.drive_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self) -> None:
        """Initialize checkpoint metadata file."""
        if not self.metadata_file.exists():
            metadata = {
                "checkpoints": [],
                "latest_generation": -1,
                "total_checkpoints": 0,
                "created_at": time.time(),
                "system_info": self._get_system_info()
            }
            self._save_metadata(metadata)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for checkpoint metadata."""
        return {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_metadata()
            return self._load_metadata()
    
    def save_checkpoint(self, state: TrainingState, generation: int) -> str:
        """
        Save compressed checkpoint with versioning.
        
        Args:
            state: Training state to save
            generation: Current generation number
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = int(time.time())
        filename = f"checkpoint_gen_{generation:06d}_{timestamp}.pkl.gz"
        filepath = self.drive_path / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'generation': generation,
            'timestamp': timestamp,
            'population': state.population,
            'model_state': state.model_state,
            'optimizer_state': state.optimizer_state,
            'metrics_history': state.metrics_history,
            'random_state': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            },
            'system_info': self._get_system_info()
        }
        
        # Save compressed checkpoint
        with gzip.open(filepath, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata["checkpoints"].append({
            "filename": filename,
            "generation": generation,
            "timestamp": timestamp,
            "file_size_mb": filepath.stat().st_size / (1024**2),
            "compression_ratio": self._calculate_compression_ratio(checkpoint_data, filepath)
        })
        metadata["latest_generation"] = generation
        metadata["total_checkpoints"] += 1
        
        self._save_metadata(metadata)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        print(f"✅ Checkpoint saved: {filename} (Gen {generation})")
        return str(filepath)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> TrainingState:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Specific checkpoint path, or None for latest
            
        Returns:
            Loaded training state
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {Path(checkpoint_path).name}")
        
        with gzip.open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Restore random states
        if 'random_state' in checkpoint_data:
            random_state = checkpoint_data['random_state']
            np.random.set_state(random_state['numpy'])
            torch.set_rng_state(random_state['torch'])
            if torch.cuda.is_available() and random_state['torch_cuda'] is not None:
                torch.cuda.set_rng_state(random_state['torch_cuda'])
        
        # Create training state
        training_state = TrainingState(
            population=checkpoint_data['population'],
            model_state=checkpoint_data['model_state'],
            optimizer_state=checkpoint_data['optimizer_state'],
            metrics_history=checkpoint_data['metrics_history'],
            random_state=checkpoint_data.get('random_state', {}),
            generation=checkpoint_data['generation'],
            timestamp=checkpoint_data['timestamp']
        )
        
        print(f"✅ Checkpoint loaded: Generation {training_state.generation}")
        return training_state
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        metadata = self._load_metadata()
        
        if not metadata["checkpoints"]:
            return None
        
        # Sort by generation (descending)
        latest = max(metadata["checkpoints"], key=lambda x: x["generation"])
        return str(self.drive_path / latest["filename"])
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        metadata = self._load_metadata()
        return [str(self.drive_path / cp["filename"]) for cp in metadata["checkpoints"]]
    
    def cleanup_old_checkpoints(self, keep_count: Optional[int] = None) -> None:
        """Remove old checkpoint files."""
        if keep_count is None:
            keep_count = self.max_versions
        
        metadata = self._load_metadata()
        checkpoints = metadata["checkpoints"]
        
        if len(checkpoints) <= keep_count:
            return
        
        # Sort by generation and keep the most recent
        checkpoints.sort(key=lambda x: x["generation"], reverse=True)
        to_remove = checkpoints[keep_count:]
        
        for checkpoint in to_remove:
            filepath = self.drive_path / checkpoint["filename"]
            if filepath.exists():
                filepath.unlink()
                print(f"🗑️ Removed old checkpoint: {checkpoint['filename']}")
        
        # Update metadata
        metadata["checkpoints"] = checkpoints[:keep_count]
        self._save_metadata(metadata)
    
    def _calculate_compression_ratio(self, data: Any, filepath: Path) -> float:
        """Calculate compression ratio."""
        try:
            uncompressed_size = len(pickle.dumps(data))
            compressed_size = filepath.stat().st_size
            return compressed_size / uncompressed_size
        except:
            return 1.0


class ColabResourceManager:
    """Resource monitoring and optimization for Google Colab."""
    
    def __init__(self, max_memory_mb: int = 12000, warning_threshold: float = 0.85):
        """
        Initialize resource manager.
        
        Args:
            max_memory_mb: Maximum memory limit in MB
            warning_threshold: Warning threshold (0-1) for resource usage
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self.monitoring_active = False
        self.monitoring_thread = None
        self.resource_history = []
    
    def monitor_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            gpu_cached = torch.cuda.memory_reserved() / (1024**2)
            total_gpu = gpu_memory + gpu_cached
        else:
            total_gpu = 0
        
        cpu_memory = psutil.virtual_memory().used / (1024**2)
        return max(total_gpu, cpu_memory)
    
    def optimize_memory(self) -> None:
        """Perform aggressive memory optimization."""
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection again
        gc.collect()
        
        print(f"🧹 Memory optimized. Current usage: {self.monitor_memory_usage():.1f} MB")
    
    def check_available_resources(self) -> Dict[str, Any]:
        """Check available computational resources."""
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2)
            }
        else:
            gpu_info = {"gpu_available": False}
        
        return {**cpu_info, **gpu_info}
    
    def estimate_resource_requirements(self, population_size: int) -> Dict[str, float]:
        """Estimate resource requirements for given population size."""
        # Rough estimates based on typical model sizes
        base_memory_mb = 500  # Base system overhead
        memory_per_individual = 10  # MB per individual
        gpu_memory_per_individual = 15 if torch.cuda.is_available() else 0
        
        estimated_memory = base_memory_mb + (population_size * memory_per_individual)
        estimated_gpu_memory = population_size * gpu_memory_per_individual
        
        return {
            "estimated_cpu_memory_mb": estimated_memory,
            "estimated_gpu_memory_mb": estimated_gpu_memory,
            "estimated_total_memory_mb": estimated_memory + estimated_gpu_memory,
            "recommended_batch_size": min(32, max(4, population_size // 4)),
            "memory_efficient": estimated_memory + estimated_gpu_memory < self.max_memory_mb * 0.8
        }
    
    def start_monitoring(self, callback: Optional[Callable] = None) -> None:
        """Start background resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    resources = self.check_available_resources()
                    current_memory = self.monitor_memory_usage()
                    
                    # Record history
                    self.resource_history.append({
                        "timestamp": time.time(),
                        "memory_usage_mb": current_memory,
                        "cpu_percent": resources["cpu_percent"],
                        "memory_percent": resources["memory_percent"]
                    })
                    
                    # Keep only last 100 records
                    if len(self.resource_history) > 100:
                        self.resource_history.pop(0)
                    
                    # Check for resource warnings
                    if current_memory > self.max_memory_mb * self.warning_threshold:
                        print(f"⚠️ High memory usage: {current_memory:.1f} MB")
                        if callback:
                            callback("memory_warning", current_memory)
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    break
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print("📊 Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        print("📊 Resource monitoring stopped")


class ColabSessionManager:
    """Manages Colab session lifecycle and timeout handling."""
    
    def __init__(self, timeout_minutes: int = 360, warning_minutes: int = 30):
        """
        Initialize session manager.
        
        Args:
            timeout_minutes: Session timeout in minutes
            warning_minutes: Warning time before timeout
        """
        self.timeout_minutes = timeout_minutes
        self.warning_minutes = warning_minutes
        self.session_start = time.time()
        self.auto_reconnect_enabled = False
        self.checkpoint_callback = None
    
    def enable_auto_reconnect(self) -> None:
        """Enable auto-reconnect functionality."""
        auto_reconnect_js = """
        function ClickConnect(){
            console.log("Working");
            document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
        }
        setInterval(ClickConnect, 60000);
        """
        
        display(Javascript(auto_reconnect_js))
        self.auto_reconnect_enabled = True
        print("🔄 Auto-reconnect enabled")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        elapsed_minutes = (time.time() - self.session_start) / 60
        remaining_minutes = self.timeout_minutes - elapsed_minutes
        
        return {
            "elapsed_minutes": elapsed_minutes,
            "remaining_minutes": max(0, remaining_minutes),
            "timeout_minutes": self.timeout_minutes,
            "session_progress": min(1.0, elapsed_minutes / self.timeout_minutes),
            "warning_threshold_reached": remaining_minutes <= self.warning_minutes
        }
    
    def set_checkpoint_callback(self, callback: Callable) -> None:
        """Set callback for emergency checkpointing."""
        self.checkpoint_callback = callback
    
    def check_timeout_warning(self) -> bool:
        """Check if timeout warning should be triggered."""
        session_info = self.get_session_info()
        
        if session_info["warning_threshold_reached"]:
            remaining = session_info["remaining_minutes"]
            print(f"⏰ Session timeout warning: {remaining:.1f} minutes remaining")
            
            # Trigger emergency checkpoint
            if self.checkpoint_callback:
                try:
                    self.checkpoint_callback()
                    print("💾 Emergency checkpoint created")
                except Exception as e:
                    print(f"❌ Emergency checkpoint failed: {e}")
            
            return True
        
        return False
    
    def create_session_widget(self) -> widgets.Widget:
        """Create session monitoring widget."""
        progress_bar = widgets.FloatProgress(
            value=0, min=0, max=1.0,
            description='Session:',
            bar_style='info',
            style={'bar_color': '#1f77b4'},
            layout=widgets.Layout(width='400px')
        )
        
        time_label = widgets.Label(value="Starting session...")
        
        def update_progress():
            session_info = self.get_session_info()
            progress_bar.value = session_info["session_progress"]
            
            remaining = session_info["remaining_minutes"]
            time_label.value = f"Time remaining: {remaining:.0f} minutes"
            
            # Change color based on remaining time
            if remaining <= self.warning_minutes:
                progress_bar.style.bar_color = '#ff7f0e'  # Orange
            if remaining <= 10:
                progress_bar.style.bar_color = '#d62728'  # Red
        
        # Update every 30 seconds
        import threading
        def update_loop():
            while True:
                update_progress()
                time.sleep(30)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
        return widgets.VBox([progress_bar, time_label])


@contextmanager
def colab_session_context(checkpoint_manager: CheckpointManager, 
                         resource_manager: ResourceManager,
                         emergency_callback: Optional[Callable] = None):
    """Context manager for robust Colab session handling."""
    session_manager = ColabSessionManager()
    
    try:
        # Setup session
        session_manager.enable_auto_reconnect()
        if emergency_callback:
            session_manager.set_checkpoint_callback(emergency_callback)
        
        resource_manager.start_monitoring()
        
        print("🚀 Colab session context initialized")
        yield session_manager
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
        if emergency_callback:
            emergency_callback()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if emergency_callback:
            emergency_callback()
        raise
    
    finally:
        # Cleanup
        resource_manager.stop_monitoring()
        print("🏁 Colab session context cleaned up")


def setup_colab_environment(drive_mount: bool = True, 
                           install_packages: bool = True) -> Dict[str, Any]:
    """
    Setup complete Colab environment for neuro-evolution.
    
    Args:
        drive_mount: Whether to mount Google Drive
        install_packages: Whether to install required packages
        
    Returns:
        Environment setup information
    """
    setup_info = {"drive_mounted": False, "packages_installed": False, "gpu_available": torch.cuda.is_available()}
    
    # Mount Google Drive
    if drive_mount:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            setup_info["drive_mounted"] = True
            print("✅ Google Drive mounted")
        except Exception as e:
            print(f"❌ Failed to mount Google Drive: {e}")
    
    # Install packages
    if install_packages:
        packages = [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "plotly>=5.0.0",
            "ipywidgets>=8.0.0",
            "psutil>=5.8.0",
            "pygad>=3.0.0"
        ]
        
        try:
            for package in packages:
                subprocess.check_call([os.sys.executable, "-m", "pip", "install", package, "-q"])
            setup_info["packages_installed"] = True
            print("✅ Required packages installed")
        except Exception as e:
            print(f"❌ Package installation failed: {e}")
    
    # Display system info
    resources = ColabResourceManager().check_available_resources()
    print(f"💻 System Info:")
    print(f"   CPU: {resources['cpu_count']} cores")
    print(f"   RAM: {resources['memory_total_gb']:.1f} GB")
    if resources.get('gpu_available'):
        print(f"   GPU: {resources['gpu_name']}")
        print(f"   GPU Memory: {resources['gpu_memory_total_gb']:.1f} GB")
    
    return setup_info

print("✅ Colab infrastructure manager implemented")