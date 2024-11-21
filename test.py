import numpy as np
import threading
from math import acos, sin
import cv2
import struct
from typing import Tuple, List

# Constants from paper
NUMBER_OF_THREADS = 8
CONFUSION_DIFFUSION_ROUNDS = 5
CONFUSION_SEED_UPPER_BOUND = 10000
CONFUSION_SEED_LOWER_BOUND = 3000
PRE_ITERATIONS = 1000
BYTES_RESERVED = 6
PI = acos(-1)

class PLCM:
    def __init__(self, control_param: float, init_condition: float):
        if control_param >= 0.5:
            control_param = 1 - control_param
        self.p = control_param
        self.x = init_condition
        self._pre_iterate()
    
    def _pre_iterate(self):
        for _ in range(PRE_ITERATIONS):
            self.x = self._single_iterate(self.x)
    
    def _single_iterate(self, x: float) -> float:
        if 0 <= x < self.p:
            return x / self.p
        elif self.p <= x <= 0.5:
            return (x - self.p) / (0.5 - self.p)
        else:
            return self._single_iterate(1.0 - x)
    
    def _double_to_bytes(self, value: float) -> bytes:
        return struct.pack('d', value)[2:2+BYTES_RESERVED]
    
    def iterate_and_get_bytes(self, iterations: int) -> Tuple[float, List[bytes]]:
        x = self.x
        byte_list = []
        
        for _ in range(iterations):
            x = self._single_iterate(x)
            byte_list.append(self._double_to_bytes(x))
            
        self.x = x
        return x, byte_list

class AssistantThread(threading.Thread):
    def __init__(self, thread_idx: int, image_shape: Tuple[int, int],
                 init_params: Tuple[Tuple[float, float], ...],
                 shared_data: dict, lock: threading.Lock):
        super().__init__()
        self.thread_idx = thread_idx
        self.height, self.width = image_shape
        self.rows_per_thread = self.height // NUMBER_OF_THREADS
        self.start_row = thread_idx * self.rows_per_thread
        self.end_row = self.start_row + self.rows_per_thread
        
        self.plcm1 = PLCM(*init_params[0])
        self.plcm2 = PLCM(*init_params[1])
        
        self.frame_ready = threading.Event()
        self.frame_processed = threading.Event()
        self.lock = lock
        self.shared_data = shared_data
        
    def confusion_operation(self):
        with self.lock:
            temp_frame = self.shared_data['temp_frame'].copy()
            confusion_seed = self.shared_data['confusion_seed']
            result_frame = self.shared_data['confused_frame']
        
        for r in range(self.start_row, self.end_row):
            for c in range(self.width):
                alpha = (r + c) % self.height
                beta = (c + int(confusion_seed * sin(2 * PI * alpha / self.height))) % self.width
                
                with self.lock:
                    result_frame[alpha, beta] = temp_frame[r, c]
                    
    def generate_byte_sequence(self) -> np.ndarray:
        pixels = (self.end_row - self.start_row) * self.width
        iterations = (pixels + BYTES_RESERVED - 1) // BYTES_RESERVED
        
        # Generate and maintain state for both PLCMs
        _, bytes1 = self.plcm1.iterate_and_get_bytes(iterations)
        _, bytes2 = self.plcm2.iterate_and_get_bytes(iterations)
        
        # Convert bytes to numpy arrays and XOR
        arr1 = np.frombuffer(b''.join(bytes1), dtype=np.uint8)
        arr2 = np.frombuffer(b''.join(bytes2), dtype=np.uint8)
        return np.bitwise_xor(arr1, arr2)[:pixels]
                    
    def diffusion_operation(self):
        byte_seq = self.generate_byte_sequence()
        
        with self.lock:
            temp_frame = self.shared_data['temp_frame'].copy()
            result_frame = self.shared_data['diffused_frame']
            diffusion_seed = self.shared_data['diffusion_seed']
        
        seq_idx = 0
        for i in range(self.start_row, self.end_row):
            for j in range(self.width):
                byte = byte_seq[seq_idx]
                
                if i == self.start_row and j == 0:
                    temp_sum = (int(temp_frame[i, j]) + byte) % 256
                    with self.lock:
                        result_frame[i, j] = byte ^ temp_sum ^ diffusion_seed
                else:
                    prev_i = i if j > 0 else i-1
                    prev_j = j-1 if j > 0 else self.width-1
                    
                    with self.lock:
                        prev_pixel = result_frame[prev_i, prev_j]
                        temp_sum = (int(temp_frame[i, j]) + byte) % 256
                        result_frame[i, j] = byte ^ temp_sum ^ prev_pixel
                
                seq_idx += 1
                
    def run(self):
        while True:
            # Confusion phase
            self.frame_ready.wait()
            self.frame_ready.clear()
            self.confusion_operation()
            self.frame_processed.set()
            
            # Diffusion phase
            self.frame_ready.wait()
            self.frame_ready.clear()
            self.diffusion_operation()
            self.frame_processed.set()

class ImageEncryptionSystem:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.uint8)
        self.height, self.width = image.shape
        self.lock = threading.Lock()
        
        np.random.seed()
        self.main_plcm = PLCM(control_param=0.37, init_condition=0.2)
        
        self.shared_data = {
            'temp_frame': np.zeros_like(image),
            'confused_frame': np.zeros_like(image),
            'diffused_frame': np.zeros_like(image),
            'confusion_seed': 0,
            'diffusion_seed': 0
        }
        
        self.threads = self._initialize_threads()
        
    def _generate_thread_parameters(self):
        params = []
        x = self.main_plcm.x
        
        for _ in range(NUMBER_OF_THREADS):
            x = self.main_plcm._single_iterate(x)
            p1 = x
            x = self.main_plcm._single_iterate(x)
            x1 = x
            x = self.main_plcm._single_iterate(x)
            p2 = x
            x = self.main_plcm._single_iterate(x)
            x2 = x
            
            params.append(((p1, x1), (p2, x2)))
        
        self.main_plcm.x = x
        return params
        
    def _initialize_threads(self):
        threads = []
        thread_params = self._generate_thread_parameters()
        
        for i in range(NUMBER_OF_THREADS):
            thread = AssistantThread(
                thread_idx=i,
                image_shape=(self.height, self.width),
                init_params=thread_params[i],
                shared_data=self.shared_data,
                lock=self.lock
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        return threads
        
    def encrypt(self) -> np.ndarray:
        # Initialize temp frame with input image
        with self.lock:
            self.shared_data['temp_frame'] = self.image.copy()
            current_frame = self.image.copy()
        
        for _ in range(CONFUSION_DIFFUSION_ROUNDS):
            # Generate confusion seed
            x = self.main_plcm._single_iterate(self.main_plcm.x)
            confusion_seed = int(abs(x) * CONFUSION_SEED_UPPER_BOUND) % \
                           (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWER_BOUND) + \
                           CONFUSION_SEED_LOWER_BOUND
            
            # Confusion phase
            with self.lock:
                self.shared_data['confusion_seed'] = confusion_seed
                self.shared_data['confused_frame'] = np.zeros_like(current_frame)
                self.shared_data['temp_frame'] = current_frame.copy()
            
            for thread in self.threads:
                thread.frame_ready.set()
            for thread in self.threads:
                thread.frame_processed.wait()
                thread.frame_processed.clear()
            
            # Update current frame after confusion
            with self.lock:
                current_frame = self.shared_data['confused_frame'].copy()
            
            # Generate diffusion seed and prepare for diffusion
            x = self.main_plcm._single_iterate(self.main_plcm.x)
            with self.lock:
                self.shared_data['diffusion_seed'] = int(x * 256) & 0xFF
                self.shared_data['diffused_frame'] = np.zeros_like(current_frame)
                self.shared_data['temp_frame'] = current_frame.copy()
            
            # Diffusion phase
            for thread in self.threads:
                thread.frame_ready.set()
            for thread in self.threads:
                thread.frame_processed.wait()
                thread.frame_processed.clear()
            
            # Update current frame after diffusion
            with self.lock:
                current_frame = self.shared_data['diffused_frame'].copy()
        
        return current_frame

def encrypt_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image")
    
    height = image.shape[0]
    if height % NUMBER_OF_THREADS != 0:
        pad_height = ((height + NUMBER_OF_THREADS - 1) // NUMBER_OF_THREADS) * NUMBER_OF_THREADS
        image = np.pad(image, ((0, pad_height - height), (0, 0)), mode='reflect')
    
    system = ImageEncryptionSystem(image)
    encrypted = system.encrypt()
    
    return image, encrypted

def main():
    #image_path = "/Users/linhha/Downloads/8-bit-256-x-256-Grayscale-Lena-Image.png"  # Replace with your image path
    image_path = "/Users/linhha/C++/PLCM/1045-2.jpg"
    original, encrypted = encrypt_image(image_path)
    
    cv2.imwrite("original.png", original)
    cv2.imwrite("encrypted.png", encrypted)
    
    
    cv2.imshow("Original", original)
    cv2.imshow("Encrypted", encrypted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()