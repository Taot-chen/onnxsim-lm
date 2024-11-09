# download pretrained model form modelscope
modelscope download 'ZhipuAI/chatglm2-6b' --cache_dir './'

# clean
mv ZhipuAI/chatglm2-6b ./
rm -rf ./ZhipuAI/ ./temp
