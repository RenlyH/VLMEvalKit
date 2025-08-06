# from http import HTTPStatus
import os
import requests
from ..dataset import DATASET_TYPE, DATASET_MODALITY
from vlmeval.api.base import BaseAPI
from vlmeval.smp import *
from ..utils import PythonInterpreter
from ..utils.python_tool import extract_tool_call_contents
import base64
from io import BytesIO
import threading
from typing import List, Any

def encode_pil_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


class ThreadSafeAppendOnlyArray:
    """Thread-safe append-only array implementation using threading.Lock"""
    
    def __init__(self):
        self._data = []
        self._lock = threading.Lock()
    
    def append(self, item):
        """Thread-safe append operation"""
        with self._lock:
            self._data.append(item)
    
    def extend(self, items):
        """Thread-safe extend operation"""
        with self._lock:
            self._data.extend(items)
    
    def get_copy(self):
        """Get a copy of the current data (thread-safe)"""
        with self._lock:
            return self._data.copy()
    
    def __len__(self):
        """Thread-safe length operation"""
        with self._lock:
            return len(self._data)
    
    def __getitem__(self, index):
        """Thread-safe item access"""
        with self._lock:
            return self._data[index]
    
    def __iter__(self):
        """Thread-safe iteration (returns copy to avoid modification during iteration)"""
        with self._lock:
            return iter(self._data.copy())

    def log_records_append_only(self, f):
        with self._lock:
            for i, item in enumerate(self._data.copy()):
                f.write(json.dumps(item) + '\n')
            # clear after writing to prevent duplicates
            self._data.clear()


class InternVL2_PromptUtil:

    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_cot = (os.getenv('USE_COT') == '1')
        use_mpo_prompt = self.use_mpo_prompt and (use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        from ..vlm.internvl.utils import (build_multi_choice_prompt,
                                          build_mcq_cot_prompt,
                                          build_qa_cot_prompt,
                                          build_mpo_prompt,
                                          reorganize_prompt)

        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))
        # TODO：support upscale_flag
        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def get_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video', 'WorldSense']  # noqa: F841
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6


class CogVLM2_PromptUtil:

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) in 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message


class LMDeployWrapper(BaseAPI):

    is_api: bool = True

    custom_prompt: str = None
    prompt_map = {
        'cogvlm2': CogVLM2_PromptUtil(),
        'internvl2': InternVL2_PromptUtil(),
        'internvl2-mpo-cot': InternVL2_PromptUtil(use_mpo_prompt=True),
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 60,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 use_tool: bool = False,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        model_id_list = [str(data['id']) for data in resp.json()['data']]
        self.model = model if model in model_id_list else model_id_list[0]
        self.logger.info(f'lmdeploy evaluate model: {self.model}')
        self.set_prompt_pattern(self.model)
        if hasattr(self, 'custom_prompt'):
            self.logger.info(f'using custom prompt {self.custom_prompt}')
        self.temperature = temperature
        self.logger.info(f'Init temperature: {self.temperature}')
        self.safe_append_array = ThreadSafeAppendOnlyArray()
        self.save_file = kwargs.get('save_file', 'saved_results.jsonl')

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].use_custom_prompt(dataset)
        return False

    def build_prompt(self, line, dataset=None):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].build_prompt(line, dataset)
        raise NotImplementedError

    def set_prompt_pattern(self, model_name):
        if 'Phi-3.5-Vision'.lower() in model_name.lower():
            self.max_tokens = 1000
            self.temperature = 0.0
        if 'cogvlm2-llama3-chat-19B'.lower() in model_name.lower():
            self.max_tokens = 2048
            self.temperature = 0.0
            self.custom_prompt = 'cogvlm2'
        if 'internvl2' in model_name.lower() or 'internvl3' in model_name.lower():
            self.max_tokens = 1024
            self.temperature = 0.0
            if 'mpo' in model_name.lower():
                self.max_tokens = 4096
                self.logger.info('Use custom prompt internvl2-mpo-cot')
                self.custom_prompt = 'internvl2-mpo-cot'
            else:
                self.logger.info('Use custom prompt internvl2')
                self.custom_prompt = 'internvl2'
        if 'internvl2-8b-mpo-cot'.lower() in model_name.lower():
            self.use_mpo_prompt = True
            self.max_tokens = 1024
            self.temperature = 0.0
            self.logger.info('Use custom prompt internvl2-mpo-cot')
            self.custom_prompt = 'internvl2-mpo-cot'
        if 'qvq'.lower() in model_name.lower():
            self.max_tokens = 4096
            self.temperature = 0.0
            self.logger.info('QVQ model detected, do not use custom prompt')

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    # Use Qwen's custom image preprocessing
                    from ..vlm.qwen2_vl.model import encode_image
                    b64, mime_type = encode_image(msg['value'])
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:{mime_type};base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if dataset is not None and listinstr(['BMMR'], dataset):
            # BMMR dataset has a very long prompt, so we need to increase max_tokens
            max_tokens = 8196
            self.logger.info('BMMR dataset detected, set max_tokens to 8196')

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()

            # for internvl2-8b-mpo-cot
            if getattr(self, 'use_mpo_prompt', False):
                from ..vlm.internvl.utils import mpo_post_processing
                answer = mpo_post_processing(answer, kwargs.get('dataset'))
        except:
            pass
        
        logging_msgs = []
        logging_msgs.append({"role": "system", "content": self.system_prompt})
        logging_msgs.append({"role": "user", "content": inputs})
        logging_msgs.append({"role": "assistant", "content": answer})
        self.safe_append_array.append(logging_msgs)
        return ret_code, answer, response


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        ret = super(LMDeployAPI, self).generate(message, dataset=dataset)
        with open(self.save_file, 'a') as f:
            self.safe_append_array.log_records_append_only(f)
        return ret
    
    def redact_images(self, inputs, placeholder='<REDACTED_IMAGE>'):
        """Redact images from inputs"""
        for msg in inputs:
            if "content" in msg and isinstance(msg['content'], list):
                for c in msg['content']:
                    if 'image' in c['type']:
                        # Redact image by removing the 'value' key
                        c['image_url'] = placeholder
            else:
                pass
        return inputs


class LMDeployAPIWithToolUse(LMDeployAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_tool = kwargs.get('use_tool', False)
        self.tool_start_token = kwargs.get('tool_start_token', None)
        self.tool_end_token = kwargs.get('tool_end_token', None)
        self.verbose = kwargs.get('verbose', False)
        if self.use_tool:
            assert self.tool_start_token and self.tool_end_token, "Both tool_start_token and tool_end_token must be provided when use_tool is True"


    def generate(self, **kwargs):
        ret = super().generate(**kwargs)

        with open(self.save_file, 'a') as f:
            for item in self.safe_append_array.get_copy():
                f.write(json.dumps(item) + '\n')
        self.safe_append_array._data.clear()  # Clear after writing to prevent duplicates
        return ret

    def setup_interpreter_with_images(self, inputs):
        """Setup interpreter with input images"""
        if not self.use_tool:
            return None

        # Create fresh interpreter instance for each generation call
        interpreter = PythonInterpreter("python", "Python code execution", {})

        # Extract PIL Images from inputs (similar to prepare_itlist)
        from PIL import Image
        images = []
        for msg in inputs:
            if msg['type'] == 'image':
                # msg['value'] is the image path
                img = Image.open(msg['value'])
                images.append(img)

        if images:
            # Reset interpreter with PIL Images
            multi_modal_data = {'image': images}
            interpreter.reset(inputs, multi_modal_data, multi_modal_data)

        return interpreter

    def generate_inner(self, inputs, **kwargs) -> str:

        if not self.use_tool:
            return super().generate_inner(inputs, **kwargs)

        # Setup interpreter with input images
        interpreter = self.setup_interpreter_with_images(inputs)
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        response_message = ""
        try_count = 0
        ret = (500, self.fail_msg, None)

        try:
            while try_count < 10:  # Limit number of rounds
                # Prepare payload with tool stop token
                payload = dict(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_tokens,
                    n=1,
                    temperature=temperature,
                    stop=[self.tool_end_token],
                    include_stop_str_in_output=True,
                    **kwargs)

                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout * 1.1)

                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

                if ret_code != 0:
                    return ret_code, self.fail_msg, response

                try:
                    resp_struct = json.loads(response.text)
                    response_message = resp_struct['choices'][0]['message']['content'].strip()
                except:
                    return ret_code, self.fail_msg, response

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})
                interpreter._log(f"Response_{interpreter.execution_count}", response_message)

                answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answers:
                    return ret_code, answers[0], response
                # Check for tool usage
                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    obs, reward, done, info = interpreter.execute(response_message)

                    content_f = []
                    content_f.append({"type": "text", "text": "<tool_response>"})
                    if isinstance(obs, dict):
                        images = obs.get('multi_modal_data', {}).get('image', [])
                        # Embed execution textual output (strip control tokens)
                        execution_text = obs['prompt']
                        # Remove system specific tokens for readability
                        execution_text = execution_text.replace("\n<|im_start|>user\n", "").replace("<|im_end|>\n<|im_start|>assistant\n", "")
                        content_f.append({"type": "text", "text": execution_text})

                        # Add captured images
                        for im in images:
                            try:
                                im_b64 = encode_pil_image_to_base64(im)
                                content_f.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im_b64}"}})
                            except Exception:
                                pass

                    elif isinstance(obs, str):
                        content_f = [{"type": "text", "text": obs}]

                    content_f.append({"type": "text", "text": "</tool_response>"})
                    input_msgs.append({"role": "user", "content": content_f})
                    # If interpreter signals completion after processing obs, return the response
                    if done:
                        return ret_code, response_message, response

                    try_count += 1
                else:
                    # No tool usage detected, return the response
                    break

            ret = (ret_code, response_message, response)
        except Exception as e:
            self.logger.error(f"Error in tool use generation: {e}")
        finally:
            # Redact images from input messages
            placeholder = '<REDACTED_IMAGE>'
            i = 0
            while i < len(inputs) and inputs[i]['type'] == 'image':
                placeholder = inputs[i]['value']
                i += 1
            self.safe_append_array.append(self.redact_images(input_msgs, placeholder=placeholder))
        
        return ret


if __name__ == '__main__':
    from unittest.mock import patch, MagicMock
    
    # Mock environment variables and create instance
    with patch.dict(os.environ, {'LMDEPLOY_API_KEY': 'test-key', 'LMDEPLOY_API_BASE': 'http://0.0.0.0:8000/v1/chat/completions'}):
        with patch('requests.get') as mock_get:
            # Mock the model list response
            mock_response = MagicMock()
            mock_response.json.return_value = {'data': [{'id': 'test-model'}]}
            mock_get.return_value = mock_response
            
            # Create instance
            api = LMDeployAPIWithToolUse(
                model='test-model',
                use_tool=True,
                tool_start_token='<python>',
                tool_end_token='</python>',
                verbose=True
            )
    
    # Test 1: Text-only input
    print("Test 1: Text-only input")
    inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    inputs = [
        {'role': 'user', 'content': [{'type': 'text', 'value': 'Hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hi there!'}]},
        {'role': 'user', 'content': [{'type': 'text', 'value': 'How are you?'}]}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 3: Text with image (simulated)
    print("Test 3: Text with image")
    inputs = [
        {'type': 'text', 'value': 'Describe this image'},
        {'type': 'image', 'value': '/path/to/image.jpg'}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 4: generate_inner with mocked response
    print("Test 4: generate_inner test")
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "The answer is 4"}}]}'
        mock_post.return_value = mock_response
        
        inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
        ret_code, answer, _ = api.generate_inner(inputs)
        print(f"Input: {inputs}")
        print(f"Return code: {ret_code}, Answer: {answer}")
    
    print("All tests completed!")


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

import math
class LMDeployAPIWithCrop(LMDeployAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_tool = kwargs.get('use_tool', False)
        self.tool_start_token = kwargs.get('tool_start_token', None)
        self.tool_end_token = kwargs.get('tool_end_token', None)
        self.verbose = kwargs.get('verbose', False)
        assert self.tool_start_token == '<tool_call>'
        if self.use_tool:
            assert self.tool_start_token and self.tool_end_token, "Both tool_start_token and tool_end_token must be provided when use_tool is True"

    def generate(self, **kwargs):
        return super().generate(**kwargs)

    def generate_inner(self, inputs, **kwargs) -> str:

        if not self.use_tool:
            return super().generate_inner(inputs, **kwargs)

        # Setup interpreter with input images
        input_msgs = self.prepare_inputs(inputs)
        for msg in inputs:
            if msg['type'] == 'image':
                # msg['value'] is the image path
                pil_img = Image.open(msg['value'])
        user_msg = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer>"
        input_msgs.append(dict(role='user', content=user_msg))

        temperature = kwargs.pop('temperature', self.temperature)
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}

        response_message = ""
        try_count = 0

        try:
            while try_count < 10:  # Limit number of rounds
                # Prepare payload with tool stop token
                payload = dict(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_tokens,
                    n=1,
                    temperature=temperature,
                    stop=[self.tool_end_token],
                    include_stop_str_in_output=True,
                    **kwargs)

                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout * 1.1)

                ret_code = response.status_code
                ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

                if ret_code != 0:
                    return ret_code, self.fail_msg, response

                try:
                    resp_struct = json.loads(response.text)
                    response_message = resp_struct['choices'][0]['message']['content'].strip()
                except:
                    return ret_code, self.fail_msg, response

                # Add assistant response to message history
                input_msgs.append({"role": "assistant", "content": response_message})

                answers = extract_tool_call_contents("<answer>", "</answer>", response_message)
                if answers:
                    return ret_code, answers, response
                # Check for tool usage
                if self.tool_start_token in response_message and self.tool_end_token in response_message:
                    action_list = response_message.split(self.tool_start_token)[1].split(self.tool_end_token)[0].strip()
                    action_list = eval(action_list)

                    bbox_list = []
                    cropped_pil_image_content_list = []

                    bbox_str = action_list['arguments']['bbox_2d']
                    bbox = bbox_str
                    left, top, right, bottom = bbox
                    cropped_image = pil_img.crop((left, top, right, bottom))
                    new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                    cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                    bbox_list.append(bbox)
                    cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                    cropped_pil_image_content_list.append(cropped_pil_image_content)

                    if len(bbox_list) == 1:
                        bbox_list = bbox_list[0]

                    content_f = []
                    content_f.append({"type": "text", "text": "<tool_response>"})
                    for cropped_pil_image_content in cropped_pil_image_content_list:
                        content_f.append(cropped_pil_image_content)
                    content_f.append({"type": "text", "text": user_msg})
                    content_f.append({"type": "text", "text": "</tool_response>"})

                    input_msgs.append({"role": "user", "content": content_f})
                    # If interpreter signals completion after processing obs, return the response

                    try_count += 1
                else:
                    # No tool usage detected, return the response
                    break

            return ret_code, response_message, response

        except Exception as e:
            self.logger.error(f"Error in tool use generation: {e}")
            return 500, self.fail_msg, None


# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


if __name__ == '__main__':
    from unittest.mock import patch, MagicMock
    
    # Mock environment variables and create instance
    with patch.dict(os.environ, {'LMDEPLOY_API_KEY': 'test-key', 'LMDEPLOY_API_BASE': 'http://0.0.0.0:8000/v1/chat/completions'}):
        with patch('requests.get') as mock_get:
            # Mock the model list response
            mock_response = MagicMock()
            mock_response.json.return_value = {'data': [{'id': 'test-model'}]}
            mock_get.return_value = mock_response
            
            # Create instance
            api = LMDeployAPIWithToolUse(
                model='test-model',
                use_tool=True,
                tool_start_token='<python>',
                tool_end_token='</python>',
                verbose=True
            )
    
    # Test 1: Text-only input
    print("Test 1: Text-only input")
    inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    inputs = [
        {'role': 'user', 'content': [{'type': 'text', 'value': 'Hello'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hi there!'}]},
        {'role': 'user', 'content': [{'type': 'text', 'value': 'How are you?'}]}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 3: Text with image (simulated)
    print("Test 3: Text with image")
    inputs = [
        {'type': 'text', 'value': 'Describe this image'},
        {'type': 'image', 'value': '/path/to/image.jpg'}
    ]
    result = api.prepare_inputs(inputs)
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    print()
    
    # Test 4: generate_inner with mocked response
    print("Test 4: generate_inner test")
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "The answer is 4"}}]}'
        mock_post.return_value = mock_response
        
        inputs = [{'type': 'text', 'value': 'What is 2+2?'}]
        ret_code, answer, _ = api.generate_inner(inputs)
        print(f"Input: {inputs}")
        print(f"Return code: {ret_code}, Answer: {answer}")
    
    print("All tests completed!")