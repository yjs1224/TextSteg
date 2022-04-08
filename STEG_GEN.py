import torch
import random
import utils
import lm
import json
import os
import jsonlines
import logging
import Huffman_Encoding

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    BartForCausalLM,
    BartTokenizer
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up
    return index


def AC_encoder(prob,bit_stream,bit_index, cur_interval, Generation_Configs):

    prob, indices = prob.sort(descending=True)
    # arithmetic coding
    cur_int_range = cur_interval[1] - cur_interval[0]  # 区间的大小  2^26
    cur_threshold = 1 / cur_int_range  # 每个区间多大
    if prob[-1] < cur_threshold:
        k = max(2, (prob < cur_threshold).nonzero()[0].item())
        prob = prob[:k]
        indices = indices[:k]

    prob = prob / prob.sum()  # 截断后线性归一化
    prob = prob.double()
    prob *= cur_int_range  # 概率转换为多少个区间
    prob = prob.round().long()  # 四舍五入取整，区间数描述的概率

    cum_probs = prob.cumsum(0)  # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
    overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]  # [299] 去掉最后一个概率
    cum_probs += cur_int_range - cum_probs[-1]  # 分布函数加到和区间数相等，区间数表示的分布函数

    cum_probs += cur_interval[0]  # 分布函数的第一项从左区间开始

    message_bits = bit_stream[bit_index: bit_index + Generation_Configs.PRECISION]  # 取了26位，但不是编码这26位，是用这26位锁定一个位置
    message_bits = [int(_) for _ in message_bits]
    message_idx = bits2int(reversed(message_bits))  # reverse只是为了计算int
    selection = (cum_probs > message_idx).nonzero()[0].item()  # 选择的单词的索引，int，选择第几个单词

    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[
        0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
    new_int_top = cum_probs[selection]

    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, Generation_Configs.PRECISION)))  # 二进制的下边界
    new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, Generation_Configs.PRECISION)))  # 二进制的上边界

    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

    new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
    new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

    cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
    cur_interval[1] = bits2int(
        reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive
    prev = indices[selection].view(1, 1)  # 一个数，代表选了哪个单词
    return cur_interval, prev, num_bits_encoded


def HC_encoder(prob, bit_stream, bit_index, Generation_Configs):
    prob, indices = prob.sort(descending=True)
    prob = prob[:2 ** Generation_Configs.bit]
    indices = indices[:2 ** Generation_Configs.bit]

    prob_dict = {i: float(p) for i, p in enumerate(prob)}
    hf = Huffman_Encoding.HuffmanCoding()
    hf.make_heap(prob_dict)
    hf.merge_nodes()
    hf.make_codes()
    for hf_code in hf.reverse_mapping.keys():
        if hf_code == bit_stream[bit_index:bit_index + len(hf_code)]:
            num_bits_encoded = len(hf_code)
            prev = indices[hf.reverse_mapping[hf_code]].view(1, 1)
            return prev, num_bits_encoded

    # old version duplicated
    # huffman coding
    # nodes = Huffman_Encoding.createNodes([_ for _ in prob])
    # root = Huffman_Encoding.createHuffmanTree(nodes)
    # codes = Huffman_Encoding.huffmanEncoding(nodes, root)
    # choose word
    # for i in range(2 ** Generation_Configs.bit):
    #     if bit_stream[bit_index:bit_index + i + 1] in codes:
    #         code_index = codes.index(bit_stream[bit_index:bit_index + i + 1])
    #         prev = indices[code_index].view(1, 1)
    #         num_bits_encoded = i+1
    #         return prev, num_bits_encoded


def ADG_encoder(prob, bit_stream, bit_index, Generation_Configs):
    prob, indices = prob.sort(descending=True)
    # start recursion
    bit_tmp = 0
    while prob[0] <= 0.5:
        # embedding bit
        bit = 1
        while (1 / 2 ** (bit + 1)) > prob[0]:
            bit += 1
        mean = 1 / 2 ** bit
        # dp
        prob = prob.tolist()
        indices = indices.tolist()
        result = []
        for i in range(2 ** bit):
            result.append([[], []])
        for i in range(2 ** bit - 1):
            result[i][0].append(prob[0])
            result[i][1].append(indices[0])
            del (prob[0])
            del (indices[0])
            while sum(result[i][0]) < mean:
                delta = mean - sum(result[i][0])
                index = near(prob, delta)
                if prob[index] - delta < delta:
                    result[i][0].append(prob[index])
                    result[i][1].append(indices[index])
                    del (prob[index])
                    del (indices[index])
                else:
                    break
            mean = sum(prob) / (2 ** bit - i - 1)
        result[2 ** bit - 1][0].extend(prob)
        result[2 ** bit - 1][1].extend(indices)
        # read secret message
        bit_embed = [int(_) for _ in bit_stream[bit_index + bit_tmp:bit_index + bit_tmp + bit]]
        int_embed = bits2int(bit_embed)
        # updating
        prob = torch.FloatTensor(result[int_embed][0]).to(device)
        indices = torch.LongTensor(result[int_embed][1]).to(device)
        prob = prob / prob.sum()
        prob, _ = prob.sort(descending=True)
        indices = indices[_]
        bit_tmp += bit

    prev = indices[int(torch.multinomial(prob, 1))].view(1,1)
    num_bits_encoded = bit_tmp
    return prev, num_bits_encoded


def main(Config):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s", device)

    Training_Configs = Config.Training
    logger.info("*****************Parser Arguments*******************")
    logger.info("Training Configs")
    logger.info(json.dumps(Training_Configs))

    if Training_Configs.model_type == "RNN":
        logger.info("Vocabulary Configs")
        Vocabulary_Configs = Config.Vocabulary
        logger.info(json.dumps(Vocabulary_Configs))
        vocabulary = utils.Vocabulary(
            Training_Configs.data_path,
            max_len=Vocabulary_Configs.MAX_LEN,
            min_len=Vocabulary_Configs.MIN_LEN,
            word_drop=Vocabulary_Configs.WORD_DROP
        )

        RNN_Configs = Config.RNN
        CELL = RNN_Configs.CELL
        EMBED_SIZE = int(RNN_Configs.EMBED_SIZE)
        HIDDEN_DIM = int(RNN_Configs.HIDDEN_DIM)
        NUM_LAYERS = int(RNN_Configs.NUM_LAYERS)
        DROPOUT_RATE = float(RNN_Configs.DROPOUT_RATE)
        logger.info("Model Configs")
        logger.info(json.dumps({**{"MODEL_TYPE":Training_Configs.model_type }, **RNN_Configs, }))
        model = lm.Old_LM(
            cell=CELL,
            vocab_size=vocabulary.vocab_size,
            embed_size=EMBED_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout_rate=DROPOUT_RATE
        )
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable params: {:d}".format(total_trainable_params))
        model.load_state_dict(torch.load(RNN_Configs.checkpoint, map_location=device))
        logger.info('checkpoint loaded')


        Generation_Configs = Config.Generation
        logger.info("Generation Configs")
        logger.info(json.dumps(Generation_Configs))

        logger.info("generating bits stream")
        logger.info("loading bits stream from cache %s"%Generation_Configs.bit_stream_file)
        with open(Generation_Configs.bit_stream_file, 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream
            bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
        bit_stream = list(bit_stream)
        random.shuffle(bit_stream)
        random.shuffle(bit_stream)
        bit_stream = ''.join(bit_stream)
        bit_index = int(torch.randint(0, high=100000, size=(1,)))

        logger.info("start generation")

        logger.info("using %s "%(Generation_Configs.alg))
        if Generation_Configs.alg.lower() == "ac":
            encoder_func = AC_encoder
        elif Generation_Configs.alg.lower() == "hc":
            encoder_func = HC_encoder
        elif Generation_Configs.alg.lower() == "adg":
            encoder_func = ADG_encoder
        else:
            logger.error("No such algorithm")
            exit()

        os.makedirs(Training_Configs.output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            with jsonlines.open(os.path.join(Training_Configs.output_dir, "stegos-encoding.jsonl"), "w") as f:
                stega_text = []
                stega_idx = 0
                while len(stega_text) < Generation_Configs.GENERATE_NUM:
                    # sample start word
                    stega_sentence = []
                    stega_bit = ['', '']
                    x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
                    samp = model.sample(x)
                    stega_sentence.append(vocabulary.i2w[samp.reshape(-1).cpu().numpy()[0]])
                    x = torch.cat([x, samp], dim=1)

                    if Generation_Configs.alg.lower() == "ac":
                        max_val = 2 **Generation_Configs.PRECISION  # num of intervals
                        cur_interval = [0, max_val]  # bottom inclusive, top exclusive
                    for i in range(Generation_Configs.MAX_GENERATE_LENGTH - 1):
                        if '_EOS' in stega_sentence:
                            break
                        # conditional probability distribution
                        log_prob = model(x)[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        prob[1] = 0
                        prob = prob / prob.sum()
                        if Generation_Configs.alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder_func(prob,bit_stream,bit_index, cur_interval,Generation_Configs)
                        elif Generation_Configs.alg.lower() == "hc":
                            prev, num_bits_encoded = encoder_func(prob, bit_stream, bit_index,Generation_Configs)
                        elif Generation_Configs.alg.lower() == "adg":
                            prev, num_bits_encoded = encoder_func(prob, bit_stream, bit_index, Generation_Configs)
                        stega_sentence.append(vocabulary.i2w[int(prev)])
                        x = torch.cat([x, prev], dim=1)
                        stega_bit.append(bit_stream[bit_index:bit_index + num_bits_encoded])
                        bit_index += num_bits_encoded
                        # early stop generation
                        if vocabulary.i2w[int(prev)] == '_EOS':
                            break

                    # check is necessray
                    if '_EOS' in stega_sentence:
                        stega_sentence.remove('_EOS')
                    stega_text.append(stega_sentence)
                    f.write({"stego": " ".join(["_BOS"] + stega_sentence + ["_EOS"]),
                             "tokens": [vocabulary.w2i[token] for token in ["_BOS"] + stega_sentence + ["_EOS"]],
                             "idx": stega_idx, "bits": stega_bit})
                    stega_idx += 1

    elif Training_Configs.model_type in ["GPT", "BART"]:
        # only CLM

        if Training_Configs.model_type == "GPT":
            LM_Configs = Config.GPT
            tokenizer = GPT2Tokenizer.from_pretrained(LM_Configs.model_name_or_path)
            model = GPT2LMHeadModel.from_pretrained(LM_Configs.model_name_or_path)
            model.to(device)
        elif Training_Configs.model_type == "BART":
            LM_Configs = Config.BART
            tokenizer = BartTokenizer.from_pretrained(LM_Configs.model_name_or_path)
            model = BartForCausalLM.from_pretrained(LM_Configs.model_name_or_path)
            model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total params: {:d}".format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable params: {:d}".format(total_trainable_params))


        Generation_Configs = Config.Generation
        logger.info("Generation Configs")
        logger.info(json.dumps(Generation_Configs))

        logger.info("generating bits stream")

        logger.info("loading bits stream from cache %s" % Generation_Configs.bit_stream_file)
        with open(Generation_Configs.bit_stream_file, 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream
            bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
            # bit_stream += bit_stream
        bit_stream = list(bit_stream)
        random.shuffle(bit_stream)
        random.shuffle(bit_stream)
        bit_stream = ''.join(bit_stream)
        bit_index = int(torch.randint(0, high=100000, size=(1,)))

        logger.info("start generation")

        logger.info("using %s " % (Generation_Configs.alg))
        if Generation_Configs.alg.lower() == "ac":
            encoder_func = AC_encoder
        elif Generation_Configs.alg.lower() == "hc":
            encoder_func = HC_encoder
        elif Generation_Configs.alg.lower() == "adg":
            encoder_func = ADG_encoder
        else:
            logger.error("No such algorithm")
            exit()

        os.makedirs(Training_Configs.output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            with jsonlines.open(os.path.join(Training_Configs.output_dir, "stegos-encoding.jsonl"), "w") as f:
                stega_text = []
                stega_idx = 0
                while len(stega_text) < Generation_Configs.GENERATE_NUM:
                    # sample start word
                    stega_sentence = []
                    # TODO begin
                    prefix = ""
                    prompt_text = Training_Configs.prompt
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + prefix + prompt_text, add_special_tokens=False,
                                                      return_tensors="pt")
                    encoded_prompt = encoded_prompt.to(device)
                    input_ids = encoded_prompt
                    stega_bit = [''] * (input_ids.shape[-1]+1)
                    logits = model(input_ids).logits[:, -1, :]
                    logits -= logits.max()
                    probs = torch.exp(logits)
                    for forbidden_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
                        probs[:, forbidden_id] = 0
                    for forbidden_id in range(256):
                        probs[:, forbidden_id] = 0
                    samp = torch.multinomial(probs,1)
                    stega_sentence.append(int(samp.view(1,1)))
                    if Training_Configs.model_type == "GPT":
                        x = torch.cat([input_ids, samp], dim=1)
                    elif Training_Configs.model_type == "BART":
                        x = torch.cat([input_ids, samp], dim=1)
                    # TODO end

                    if Generation_Configs.alg.lower() == "ac":
                        max_val = 2 ** Generation_Configs.PRECISION  # num of intervals
                        cur_interval = [0, max_val]  # bottom inclusive, top exclusive
                    for i in range(Generation_Configs.MAX_GENERATE_LENGTH - 1):
                        if '_EOS' in stega_sentence:
                            break
                        # conditional probability distribution
                        # todo begin
                        log_prob = model(x).logits[:, -1, :]
                        log_prob -= log_prob.max()
                        prob = torch.exp(log_prob).reshape(-1)
                        if Training_Configs.model_type == "BART":
                            prob[tokenizer.unk_token_id] = 0
                        for forbidden_id in range(256):
                            prob[forbidden_id] = 0
                        # todo end
                        prob = prob / prob.sum()
                        # print(prob[tokenizer.eos_token_id])
                        # if prob.argmax() == tokenizer.eos_token_id:
                        #     break
                        if Generation_Configs.alg.lower() == "ac":
                            cur_interval, prev, num_bits_encoded = encoder_func(prob, bit_stream, bit_index,
                                                                                cur_interval, Generation_Configs)
                        elif Generation_Configs.alg.lower() == "hc":
                            prev, num_bits_encoded = encoder_func(prob, bit_stream, bit_index, Generation_Configs)
                        elif Generation_Configs.alg.lower() == "adg":
                            prev, num_bits_encoded = encoder_func(prob, bit_stream, bit_index, Generation_Configs)
                        # early stop generation
                        if int(prev) == tokenizer.eos_token_id:
                            break
                        stega_sentence.append(int(prev))
                        x = torch.cat([x, prev], dim=1)
                        stega_bit.append(bit_stream[bit_index:bit_index + num_bits_encoded])
                        bit_index += num_bits_encoded
                        # # early stop generation
                        # if int(prev) == tokenizer.eos_token_id:
                        #     break

                    # check is necessray
                    if tokenizer.eos_token_id in stega_sentence:
                        stega_sentence.remove(tokenizer.eos_token_id)
                    stega_text.append(tokenizer.decode(stega_sentence))
                    f.write({"stego": "_BOS " + tokenizer.decode(stega_sentence) + " _EOS",
                             "tokens": stega_sentence ,
                             "idx": stega_idx, "bits": stega_bit})
                    stega_idx += 1
    logger.info("finished generation")


if __name__ == '__main__':
    import argparse
    # t = T5Tokenizer.from_pretrained("t5-base")
    parser = argparse.ArgumentParser(description="argument for generation")
    parser.add_argument("--config_path", type=str, default="./Configs/commonsense-gpt-ac.json")
    args = parser.parse_args()
    Config = utils.Config(args.config_path).get_configs()
    main(Config)