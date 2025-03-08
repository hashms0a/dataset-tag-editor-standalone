from typing import TYPE_CHECKING, Callable
import gradio as gr
from PIL import Image
from .ui_common import *
from .uibase import UIBase
import os

if TYPE_CHECKING:
    from .ui_classes import *

Filter = dte_module.filters.Filter

class DatasetGalleryUI(UIBase):
    def __init__(self):
        self.selected_path = ""
        self.selected_index = -1

    def create_ui(self, image_columns: int, get_filters: Callable[[], list[Filter]]):
        self.gl_dataset_images = gr.Gallery(
            label="Dataset Images", elem_id="dataset_gallery", columns=image_columns, preview=True)
        self.get_filters = get_filters

    def set_callbacks(self, gallery_state: 'GalleryStateUI'):
        
        def gl_dataset_images_on_change(select_data: gr.SelectData):
            imgs = dte_instance.get_filtered_imgpaths(self.get_filters())
            valid_imgs = [img for img in imgs if os.path.isfile(img)]
            if select_data.selected and 0 <= select_data.index < len(valid_imgs):
                self.selected_index = select_data.index
                self.selected_path = valid_imgs[self.selected_index]
            else:
                self.selected_index = -1
                self.selected_path = ""
            gallery_state.register_value("Selected Image", self.selected_path)

        def load_images_to_gallery():
            imgs = dte_instance.get_filtered_imgpaths(self.get_filters())
            valid_imgs = [img for img in imgs if os.path.isfile(img)]
            return [[Image.open(img), os.path.basename(img)] for img in valid_imgs]

        # Assigning callbacks
        self.gl_dataset_images.select(gl_dataset_images_on_change)
        
        # Trigger loading images when needed, e.g., on an event or button click.
        load_button = gr.Button("Load Images")
        load_button.click(fn=load_images_to_gallery, inputs=[], outputs=self.gl_dataset_images)
