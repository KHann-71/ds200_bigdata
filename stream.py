import socket
import pickle
import json
import time
import argparse
import numpy as np
import os

def load_cifar_batches(folder, is_train=True):
    files = [f"data_batch_{i}" for i in range(1, 6)] if is_train else ["test_batch"]
    return [os.path.join(folder, f) for f in files]

def read_batch(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding='bytes')
    X = data[b'data']
    y = data[b'labels']
    return X, y

def main(args):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.host, args.port))
    s.listen(1)
    print(f"📡 Waiting for Spark to connect on {args.port}...")
    conn, addr = s.accept()
    print(f"✅ Connected to {addr}")

    batch_files = load_cifar_batches(args.folder, args.train)
    for file in batch_files:
        X, y = read_batch(file)
        for i in range(0, len(X), args.batch_size):
            batch_X = X[i:i+args.batch_size]
            batch_y = y[i:i+args.batch_size]
            payload = {
                idx: {f"feature-{j}": float(val) for j, val in enumerate(img)}
                     | {"label": int(label)}
                for idx, (img, label) in enumerate(zip(batch_X, batch_y))
            }
            conn.send((json.dumps(payload) + "\n").encode())
            time.sleep(args.sleep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sleep", type=int, default=2)
    parser.add_argument("--port", type=int, default=6100)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    main(args)
