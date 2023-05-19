import modules.scripts as scripts
import gradio as gr
import os
import json

from modules import script_callbacks
from scripts.dream import start_dreamer


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            create_basic_tab()

        return [(ui_component, "Headtrip", "HeadtripTab")]
    


def load_imgnet_classes():
    """
    load classes from json for dropdown

    """
    current_dir = os.getcwd()
    relative_path = os.path.join("extensions","HeadTrip-Extension","scripts")
    imgnet_path = os.path.join(current_dir, relative_path, "imageNet_classes.json")
    try:
        with open(imgnet_path, 'r') as openfile:
            return json.load(openfile)

    except Exception as e:
        print(f"Exception loading config: {e}")
        return None


def create_basic_tab():
    with gr.Column(visible=True, elem_id="Processid") as second_panel:
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            with gr.Tabs(elem_id="input_headtrip"):
                with gr.Row():
                    with gr.Tab(elem_id="input_diffuse", label="Image"):
                        with gr.Column():
                            with gr.Row():
                                input_image = gr.Image(
                                    label="Input_Image", elem_id="input_page2"
                                )

                dummy_component = gr.Label(visible=False)
                with gr.Row():
                    with gr.Column():
                        num_iterations = gr.Slider(
                            minimum=1.0,
                            maximum=200.0,
                            step=1.0,
                            value=10.0,
                            label="Iterations",
                        )
                        num_octaves = gr.Slider(
                            minimum=1.0,
                            maximum=100.0,
                            step=1.0,
                            value=20.0,
                            label="Number Octaves",
                        )

                    with gr.Column():
                        octave_scale = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            step=0.1,
                            value=1.2,
                            label="Octave Scale",
                        )
                        lr = gr.Number(
                            value=0.008,
                            label="Learning Rate",
                            interactive=True,
                        )
                dummy_component = gr.Label(visible=False)
                
                with gr.Row():
                    with gr.Column():
                        model = gr.Dropdown(choices=["vgg19", "resnet", "densenet"], value="vgg19", label="Model")
                    with gr.Column():
                        layer = gr.Dropdown(choices=[str(i) for i in range(0, 27)], value="26", label="Layer")

                with gr.Row():
                    with gr.Column():
                        single_class = gr.Checkbox(value=True, label="Guided Dreaming", info="Uses only on class")
                    with gr.Column():
                        labels = load_imgnet_classes()
                        inv_labels = {v: k for k, v in labels.items()}
                        m_class = gr.Dropdown(choices=list(inv_labels.keys()), value=labels["0"], label="Classes")

                with gr.Row():
                    max_class = gr.Checkbox(value=False, label="Maximum Class Output", info="Adds only the most likely class on the image, Guided Dreaming must be checked")

            with gr.Tabs(elem_id="result_headtrip"):
                with gr.TabItem(elem_id="output_headtrip", label="Output"):
                    with gr.Row():
                        result_image = gr.outputs.Image(type="pil")
                        result_image.style(height=400, width=400)
                    with gr.Row():
                        runbutton = gr.Button("Dream")

    runbutton.click(dream, [input_image, num_iterations, num_octaves, octave_scale, lr, model, m_class, single_class, max_class, layer], result_image)


def dream(input_image, num_iterations, num_octaves, octave_scale, lr, model, m_class, single_class, max_class, layer):

    labels = load_imgnet_classes()
    inv_labels = {v: k for k, v in labels.items()}

    params = {
        "guided" : single_class,
        "max_output" : max_class,
        "pyramid_max" : True,
        "channel_list": [int(inv_labels[m_class])],
        "no_class": not single_class,
        "at_layer": int(layer),
        "use_spynet": False,
        "use_depth": False,
        "depth_str":1.2,
        "use_threshold": True,
        "th_val": 0.2,
        "invert_depth": False,
        "pretrained": True,
        "start_position": 0,
        "seq": False,
        "fp16": True,
        "random": False }
    
    params['num_iterations'] = num_iterations
    params['num_octaves'] = num_octaves
    params['octave_scale'] = octave_scale
    params['model'] = model
    params['lr'] = lr


    output = start_dreamer(input_image, params)
    return output

script_callbacks.on_ui_tabs(on_ui_tabs)
