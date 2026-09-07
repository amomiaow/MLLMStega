import json
import os
import random
import re
from http import HTTPStatus

import dashscope
import numpy as np
import requests
import torch
from dashscope import MultiModalConversation
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

# ---------------------------------------------------------------------------
# Local model paths
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
VL_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen2.5-VL")  # Qwen2.5-VL-7B-Instruct (local)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-xxx")  # DashScope key for image generation
# DashScope endpoint used for image generation.
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://llm-859yb3v6ip00soci.cn-beijing.maas.aliyuncs.com/api/v1",
)
dashscope.base_http_api_url = DASHSCOPE_BASE_URL


# System prompt used to generate the multi-dimensional feature option set.
PROMPT_ENGINEER_SYSTEM_PROMPT = (
    "#Role: image generation prompt engineer\n"
    "You are a professional image generation prompt engineer who specializes in "
    "breaking down themes into mutually exclusive explicit visual dimensions.\n"
    "## Rules:\n"
    "1. Dimension Limitations\n"
    "- The dimension range: the salient features which can have rich options, accessories, the pose, "
    "the background, decorative elements.\n"
    "- Dimension describing color must avoid describing the overall color of the image and "
    "instead indicate the color of specific objects/parts of the image. For example: Hair color.\n"
    "- Forbidden dimensions: Expression, Lighting, Atmosphere.\n"
    "2. Option Limitations\n"
    "- There must be a very large visual difference between options of the same dimension.\n"
    "- Options between different dimensions cannot conflict or duplicate.\n"
    "3. Strict Quantity Requirements\n"
    "Exactly 6 dimensions, each dimension have exactly 8 options.\n"
    "## Output format:\n"
    "{\n"
    "  \"dimension_1_name\": [\"option1\", \"option2\", ..., \"option8\"],\n"
    "  \"dimension_2_name\": [\"option1\", \"option2\", ..., \"option8\"],\n"
    "  // ... exactly 6 dimensions\n"
    "}\n"
    "## Additional Instructions:\n"
    "- Output ONLY valid JSON without any additional text\n"
    "- Use double quotes for keys and string values\n"
    "- Ensure all 8 options per dimension are distinct and visually diverse"
)


def set_seed(seed):
    """Set the random seed across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_to_matrix(json_str):
    """Convert a JSON string into a matrix."""
    data = json.loads(json_str)

    # Get the dimensions and options.
    dimensions = list(data.keys())
    options_matrix = []

    # Build the matrix.
    for dim in dimensions:
        options_matrix.append(data[dim])

    return dimensions, np.array(options_matrix)


def matrix_to_json(dimensions, matrix):
    """Convert a matrix back into a JSON-formatted string."""
    # Convert the matrix into a dictionary.
    data = {}
    for i, dim in enumerate(dimensions):
        # Handle numpy data types (e.g. np.int32, np.float64) in the matrix.
        options = matrix[i].tolist() if isinstance(matrix[i], np.ndarray) else matrix[i]

        # Ensure all elements are native Python types (for JSON serialization).
        data[dim] = [item.item() if isinstance(item, np.generic) else item for item in options]

    # Convert to a JSON string.
    return json.dumps(data, ensure_ascii=False)


def permute_matrix(dimensions, matrix, seed=None):
    """Apply row (dimension) and column (option) permutations to the matrix."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Row permutation (dimension permutation).
    row_order = list(range(len(dimensions)))
    random.shuffle(row_order)
    permuted_dimensions = [dimensions[i] for i in row_order]
    row_permuted_matrix = matrix[row_order, :]

    # Column permutation (option permutation).
    col_permuted_matrix = np.copy(row_permuted_matrix)
    for i in range(col_permuted_matrix.shape[0]):
        col_order = list(range(col_permuted_matrix.shape[1]))
        random.shuffle(col_order)
        col_permuted_matrix[i] = col_permuted_matrix[i, col_order]

    return permuted_dimensions, col_permuted_matrix


def generate_binary_string(length=18, seed=None):
    """Generate a random binary string of the given length."""
    if seed is not None:
        random.seed(seed)

    return ''.join(str(random.randint(0, 1)) for _ in range(length))


def split_binary_string(binary_str, segment_length=3):
    """Split a binary string into segments of the given length."""
    segments = []
    for i in range(0, len(binary_str), segment_length):
        segment = binary_str[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    return segments


def binary_to_index(binary_str):
    """Convert a 3-bit binary string into an index (0-7)."""
    return int(binary_str, 2)


def select_options(dimensions, matrix, binary_segments):
    """Select one option per dimension according to the binary segments."""
    selected = {}
    for i, segment in enumerate(binary_segments):
        if i < len(dimensions):  # Do not exceed the number of dimensions.
            dim = dimensions[i]
            index = binary_to_index(segment)

            # Keep the index within the valid range.
            if index < matrix.shape[1]:
                selected[dim] = matrix[i, index]
            else:
                # Fall back to the last option if the index is out of range.
                selected[dim] = matrix[i, -1]
    return selected


def select_all_except_index(dimensions, matrix, binary_segments):
    """Select every option in each dimension except the one at the given index."""
    selected = {}
    for i, segment in enumerate(binary_segments):
        if i < len(dimensions):  # Do not exceed the number of dimensions.
            dim = dimensions[i]
            index = binary_to_index(segment)

            # Get all options of the current dimension.
            all_options = matrix[i]

            # Keep the index within the valid range.
            if index >= len(all_options):
                # Fall back to excluding the last option if the index is out of range.
                index = len(all_options) - 1

            # Select every option except the one at the index.
            selected[dim] = [option for j, option in enumerate(all_options) if j != index]

    return selected


def print_matrix(title, dimensions, matrix):
    """Print the matrix."""
    print(f"\n{title}")
    print("-" * 80)

    # Print the header.
    header = "Dimension".ljust(20)
    for i in range(1, matrix.shape[1] + 1):
        header += f"Option {i}".ljust(20)
    print(header)
    print("-" * 80)

    # Print each row.
    for i, dim in enumerate(dimensions):
        row_str = dim.ljust(20)
        for option in matrix[i]:
            row_str += str(option).ljust(20)
        print(row_str)

    print("-" * 80)


def permuted_matrix_to_text(permuted_dims, permuted_matrix):
    """Convert the permuted matrix into a text description."""
    text = ("This is an AI-generated image composed of one option selected from each of the following dimensions. "
            "Please review each dimension in its entirety. Select the option used to generate it in each dimension. "
            "Each dimension has only one option. Answer in JSON format.\n")

    for i, dim in enumerate(permuted_dims):
        options = permuted_matrix[i]
        text += f"{dim}: {', '.join(options)}\n"

    return text


def extract_json(text):
    """Strip markdown code fences and extract the raw JSON from the model output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.DOTALL)
    return text


# ---------------------------------------------------------------------------
# Local inference helpers
# ---------------------------------------------------------------------------
def load_vl_model(path, device="cuda:0"):
    """Load the local Qwen2.5-VL model and its processor (4-bit quantized).
    """
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, quantization_config=quantization_config, device_map=device
    )
    processor = AutoProcessor.from_pretrained(path)
    model.eval()
    return model, processor


def llm_generate(model, processor, system_prompt, user_prompt, seed, max_new_tokens=2048):
    """Run the local Qwen2.5-VL model in text-only mode (no image input)."""
    set_seed(seed)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=generation_config)

    output_ids_trimmed = [
        out[len(inp):] for out, inp in zip(output_ids, inputs.input_ids)
    ]
    return processor.batch_decode(
        output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def analyze_image(model, processor, image_path, prompt_text, max_new_tokens=2048):
    """Run the local Qwen2.5-VL model with an image + text input."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": os.path.abspath(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_ids_trimmed = [
        out[len(inp):] for out, inp in zip(output_ids, inputs.input_ids)
    ]
    return processor.batch_decode(
        output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def generate_image(prompt, negative_prompt, seed, size="1024*1024"):
    """Generate an image with the DashScope qwen-image-2.0 API (sync call).

    The generated image is downloaded and saved to ./generated.png so the local
    VL model can analyze it during the recovery phase.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }
    ]
    rsp = MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-image-2.0",
        messages=messages,
        result_format="message",
        stream=False,
        n=1,
        watermark=False,
        prompt_extend=True,
        negative_prompt=negative_prompt,
        seed=seed,
        size=size,
    )
    print('response: %s' % rsp)
    if rsp.status_code != HTTPStatus.OK:
        print('image gen Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))
        return None

    # The generated image URL(s) are returned in choices[0].message.content as a
    # list of {"image": "url"} / {"text": "..."} entries.
    content = rsp.output.choices[0].message.content
    image_url = None
    for item in content:
        if isinstance(item, dict) and "image" in item:
            image_url = item["image"]
            break
    if not image_url:
        print('no image url found in response content: %s' % content)
        return None

    image_path = os.path.abspath('./generated.png')
    with open(image_path, 'wb+') as f:
        f.write(requests.get(image_url).content)
    return image_path


if __name__ == "__main__":

    theme = 'woman'
    key = 1234

    # Load the local VL model.
    vl_model, vl_processor = load_vl_model(VL_MODEL_PATH, device="cuda:0")

    # Hide phase.
    secret = '010000101011110100'

    # 1. Generate the multi-dimensional feature option set.
    theme_set = llm_generate(
        vl_model, vl_processor,
        PROMPT_ENGINEER_SYSTEM_PROMPT,
        "Theme: " + theme,
        seed=key,
    )
    print(theme_set)
    dimensions, matrix = json_to_matrix(extract_json(theme_set))
    print_matrix("Multi-dimensional feature option set", dimensions, matrix)

    # 2. Apply row and column permutations.
    permuted_dims, permuted_matrix = permute_matrix(dimensions, matrix, key)
    codebook = matrix_to_json(permuted_dims, permuted_matrix)
    print_matrix("Permuted matrix", permuted_dims, permuted_matrix)

    # 3. Split the binary string into 6 segments of 3 bits each.
    segments = split_binary_string(secret, 3)
    print(segments)

    # 4. Select options according to the binary segments.
    selection = select_options(permuted_dims, permuted_matrix, segments)
    prompt = theme + ', ' + str(selection)
    negative_prompt = str(select_all_except_index(permuted_dims, permuted_matrix, segments))
    print("\nFinal selection:")
    print(selection)

    input()

    # 5. Generate the image via the DashScope API (the VL model stays resident).
    print('----generating image via API, please wait a moment----')
    image_path = generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=key,
    )
    print('saved image to: %s' % image_path)

    input()

    # Recovery phase.
    # 6. Reproduce the multi-dimensional feature option set.
    re_theme_set = llm_generate(
        vl_model, vl_processor,
        PROMPT_ENGINEER_SYSTEM_PROMPT,
        "Theme: " + theme,
        seed=key,
    )
    re_dimensions, re_matrix = json_to_matrix(extract_json(re_theme_set))
    print_matrix("Multi-dimensional feature option set", re_dimensions, re_matrix)

    # 7. Reproduce the codebook.
    re_permuted_dims, re_permuted_matrix = permute_matrix(re_dimensions, re_matrix, key)
    print_matrix("Permuted matrix", re_permuted_dims, re_permuted_matrix)

    input()

    # 8. Parse the features from the generated image.
    prompt_text = permuted_matrix_to_text(re_permuted_dims, re_permuted_matrix)
    decide = analyze_image(vl_model, vl_processor, image_path, prompt_text)
    decide = extract_json(decide)
    print(decide)

    input()

    # 9. Recover the secret message.
    options = json.loads(decide)
    codebook = json.loads(matrix_to_json(re_permuted_dims, re_permuted_matrix))
    sequence = []
    for i, j in options.items():
        index = codebook[i].index(j)  # Get the index of the option in the list.
        sequence.append(index)

    # Convert to 3-bit binary and concatenate.
    binary_str = ''.join(format(index, '03b') for index in sequence)

    print("Original secret:", secret)
    print("Recovered secret:", binary_str)
