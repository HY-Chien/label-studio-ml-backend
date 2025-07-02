from typing import List, Dict, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from ultralytics import YOLO
import os
import json


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")
        self.model = YOLO("/home/S113065528/Yulon-AIAMS-label-studio/checkpoints/icons=Cv4_ds=20k(8,1,1)_epoch=60_yolo12m/weights/best.pt")
        print(f"[INFO] Model classes: {self.model.names}")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        print(f"[INFO] Predicting on {len(tasks)} tasks...")
        predictions = []

        for task in tasks:
            image_url = task["data"]["image"]
            image_path = self.get_local_path(image_url, task_id=task["id"])
            print(f"[DEBUG] local image path: {image_path}")

            result = self.model(image_path)[0]
            print(f"[DEBUG] model result: {result.boxes}")

            image_w, image_h = result.orig_shape[1], result.orig_shape[0]
            result_items = []

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.model.names[label]

                result_items.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [class_name],
                        "x": x1 * 100 / image_w,
                        "y": y1 * 100 / image_h,
                        "width": (x2 - x1) * 100 / image_w,
                        "height": (y2 - y1) * 100 / image_h
                    },
                    "score": conf
                })

            predictions.append({
                "result": result_items,
                "model_version": self.get("model_version")
            })

        print(f"[DEBUG] tasks: {tasks}")
        print("[DEBUG] Final ModelResponse:", json.dumps(ModelResponse(predictions=predictions).dict(), indent=2))
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training

        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check Webhook event reference)
        """
        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')

        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')

        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')
        print('fit() completed successfully.')
