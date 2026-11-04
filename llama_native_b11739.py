import ctypes as C
import os
from pathlib import Path
import time

P=C.c_void_p
I=C.c_int32
U=C.c_uint32
F=C.c_float
B=C.c_bool

class ModelParams(C.Structure):
    _fields_=[('devices',P),('tensor_buft_overrides',P),('n_gpu_layers',I),
              ('split_mode',I),('load_mode',I),('lazy_mode',I),('main_gpu',I),
              ('tensor_split',P),('progress_callback',P),('progress_callback_user_data',P),
              ('kv_overrides',P)]+[(n,B) for n in ['vocab_only','check_tensors',
                 'use_extra_bufts','no_host','no_alloc','load_mtp']]

class ContextParams(C.Structure):
    _fields_=[(n,U) for n in ['n_ctx','n_batch','n_ubatch','n_seq_max','n_rs_seq',
                'n_outputs_max','n_outputs_max_per_seq']]+[(n,I) for n in [
                'n_threads','n_threads_batch','ctx_type','rope_scaling_type',
                'pooling_type','attention_type','flash_attn_type']]+[(n,F) for n in [
                'rope_freq_base','rope_freq_scale','yarn_ext_factor','yarn_attn_factor',
                'yarn_beta_fast','yarn_beta_slow']]+[('yarn_orig_ctx',U),('defrag_thold',F),
                ('cb_eval',P),('cb_eval_user_data',P),('type_k',I),('type_v',I),
                ('moe_cache_size',C.c_size_t),('abort_callback',P),('abort_callback_data',P)
                ]+[(n,B) for n in ['embeddings','offload_kqv','no_perf','op_offload',
                'swa_full','kv_unified']]+[('samplers',P),('n_samplers',C.c_size_t),('ctx_other',P)]

class Batch(C.Structure):
    _fields_=[('n_tokens',I)]+[(n,P) for n in ['token','embd','pos','n_seq_id','seq_id','logits']]

def bind(lib,name,result,args):
    fn=getattr(lib,name);fn.restype=result;fn.argtypes=args
    return fn

class NativeCPU:
    def __init__(self,runtime,model,threads=8,context=2048):
        self.runtime=Path(runtime).resolve()
        self.dll_directory=os.add_dll_directory(str(self.runtime))
        self.ggml=C.CDLL(str(self.runtime/'ggml.dll'))
        bind(self.ggml,'ggml_backend_load_all_from_path',None,[C.c_char_p])(b'runtime')
        self.lib=C.CDLL(str(self.runtime/'llama.dll'))
        l=self.lib
        bind(l,'llama_backend_init',None,[])()
        bind(l,'llama_model_default_params',ModelParams,[])
        bind(l,'llama_context_default_params',ContextParams,[])
        bind(l,'llama_model_load_from_file',P,[C.c_char_p,ModelParams])
        bind(l,'llama_init_from_model',P,[P,ContextParams])
        bind(l,'llama_model_get_vocab',P,[P])
        bind(l,'llama_get_memory',P,[P])
        bind(l,'llama_memory_clear',None,[P,B])
        bind(l,'llama_batch_get_one',Batch,[C.POINTER(I),I])
        bind(l,'llama_decode',I,[P,Batch])
        bind(l,'llama_sampler_init_greedy',P,[])
        bind(l,'llama_sampler_sample',I,[P,P,I])
        bind(l,'llama_sampler_accept',None,[P,I])
        bind(l,'llama_sampler_free',None,[P])
        bind(l,'llama_vocab_is_eog',B,[P,I])
        bind(l,'llama_free',None,[P])
        bind(l,'llama_model_free',None,[P])
        mp=l.llama_model_default_params();mp.n_gpu_layers=0
        self.model=l.llama_model_load_from_file(str(model).encode('utf8'),mp)
        if not self.model:raise RuntimeError('Native model load failed')
        cp=l.llama_context_default_params();cp.n_ctx=context
        cp.n_batch=512;cp.n_ubatch=128;cp.n_seq_max=1
        cp.n_threads=cp.n_threads_batch=threads
        cp.offload_kqv=False;cp.op_offload=False;cp.no_perf=False
        self.ctx=l.llama_init_from_model(self.model,cp)
        if not self.ctx:raise RuntimeError('Native context creation failed')
        self.vocab=l.llama_model_get_vocab(self.model)

    def generate(self,input_ids,max_tokens=128):
        l=self.lib
        l.llama_memory_clear(l.llama_get_memory(self.ctx),True)
        sampler=l.llama_sampler_init_greedy()
        start=time.monotonic()
        try:
            # Last-prompt-token logits are returned by batch_get_one.
            prompt=(I*len(input_ids))(*input_ids)
            code=l.llama_decode(self.ctx,l.llama_batch_get_one(prompt,len(input_ids)))
            if code:raise RuntimeError('Prompt decode failed '+str(code))
            prompt_seconds=time.monotonic()-start
            output=[];stop='length'
            for _ in range(max_tokens):
                token=int(l.llama_sampler_sample(sampler,self.ctx,-1))
                l.llama_sampler_accept(sampler,token)
                if l.llama_vocab_is_eog(self.vocab,token):
                    stop='stop';break
                output.append(token)
                if len(output)==max_tokens:break
                one=(I*1)(token)
                code=l.llama_decode(self.ctx,l.llama_batch_get_one(one,1))
                if code:raise RuntimeError('Generation decode failed '+str(code))
            return output,stop,{'prompt_seconds':prompt_seconds,
                               'total_seconds':time.monotonic()-start}
        finally:l.llama_sampler_free(sampler)

    def close(self):
        if self.ctx:self.lib.llama_free(self.ctx);self.ctx=None
        if self.model:self.lib.llama_model_free(self.model);self.model=None
