from pydantic import BaseModel, Field, ConfigDict
from typing import List

class TransactionInput(BaseModel):
    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in Euros")
    v_features: List[float] = Field(
        ..., 
        min_length=28, 
        max_length=28, 
        description="List of 28 anonymized PCA feature values V1 to V28"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 0.0,
                "Amount": 149.62,
                "v_features": [
                    -1.359807, -0.072781, 2.536347, 1.378155,
                    -0.338321, 0.462388, 0.239599, 0.098698,
                    0.363787, 0.090794, -0.551600, -0.617801,
                    -0.991390, -0.311169, 1.468177, -0.470401,
                    0.207971, 0.025791, 0.403993, 0.251412,
                    -0.018307, 0.277838, -0.110474, 0.066928,
                    0.128539, -0.189115, 0.133558, -0.021053
                ]
            }
        }
    )
