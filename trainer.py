from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import VectorUDT
from dataloader import DataLoader
from transforms import Transforms
from typing import Any
import datetime

class Trainer:
    def __init__(self, model: Any, host: str, port: int, transforms: Transforms):
        self.model = model
        self.sc = SparkContext(appName="BigDataStreamingML")
        self.ssc = StreamingContext(self.sc, batchDuration=3)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc, self.ssc, self.sqlContext, host, port, transforms)

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self._process_batch)
        self.ssc.start()
        self.ssc.awaitTermination()

    def _process_batch(self, time, rdd):
        if rdd.isEmpty():
            print(f"[{time}] Empty RDD")
            return

        schema = StructType([
            StructField("image", VectorUDT(), True),
            StructField("label", IntegerType(), True)
        ])

        df = self.sqlContext.createDataFrame(rdd, schema)
        preds, acc, prec, rec, f1 = self.model.train(df)

        print(f"\n[{time}] 📊 Training Stats:")
        print(f"  ✅ Accuracy: {acc:.4f}")
        print(f"  🎯 Precision: {prec:.4f}")
        print(f"  🔁 Recall: {rec:.4f}")
        print(f"  💯 F1 Score: {f1:.4f}")
