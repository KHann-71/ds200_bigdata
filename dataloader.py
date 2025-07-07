import numpy as np
import json
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
from transforms import Transforms

class DataLoader:
    def __init__(self, sc: SparkContext, ssc: StreamingContext, sqlContext: SQLContext,
                 host: str, port: int, transforms: Transforms):
        self.sc = sc
        self.ssc = ssc
        self.sqlContext = sqlContext
        self.transforms = transforms
        self.stream = ssc.socketTextStream(hostname=host, port=port)

    def parse_stream(self) -> DStream:
        return (
            self.stream
            .map(json.loads)
            .flatMap(lambda batch: batch.values())
            .map(lambda record: (np.array(list(record.values())[:-1]).reshape(3, 32, 32)
                                 .transpose(1, 2, 0).astype(np.uint8),
                                 record["label"]))
            .map(lambda x: [self.transforms.transform(x[0]).reshape(-1).tolist(), x[1]])
            .map(lambda x: [DenseVector(x[0]), int(x[1])])
        )
