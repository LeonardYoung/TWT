import torch.nn as nn
import torch
import config as Config
import torch.nn.functional as F
import demo_model

def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class TimeSeriesTransformer(nn.Module):
    def __init__(
            self,
            n_encoder_inputs=2,
            n_decoder_inputs=2,
            channels=16,
            out_dim=10,
            dropout=0.1,
            lr=1e-4,
    ):
        super().__init__()


        if Config.transformer_of_torch:

            self.lr = lr
            self.dropout = dropout

            self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)
            self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=Config.n_head,
                dropout=self.dropout,
                dim_feedforward=4 * channels,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=channels,
                nhead=Config.n_head,
                dropout=self.dropout,
                dim_feedforward=4 * channels,
            )

            if Config.swap_dimen:
                n_encoder_inputs = Config.seq_len_x
                n_decoder_inputs = Config.seq_len_forcast
                out_dim = Config.seq_len_forcast

            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=Config.num_layers)
            self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=Config.num_layers)

            self.input_projection = nn.Linear(n_encoder_inputs, channels)
            self.output_projection = nn.Linear(n_decoder_inputs, channels)

            self.linear = nn.Linear(channels, out_dim)
            if Config.encoder_only:
                self.encoder_out_linear = nn.Linear(24, 8)

            self.do = nn.Dropout(p=self.dropout)
        else:
            self.encoder = demo_model.Encoder(vocab_size=1024,d_model=channels,d_ff=channels * 4,
                                              d_k=channels * 4,d_v=channels * 4,n_heads=8,
                                              n_layers=Config.num_layers,pad_index=0,device=Config.device)
            self.decoder = demo_model.Decoder(vocab_size=1024,d_model=channels,d_ff=channels * 4,
                                              d_k=channels * 4,d_v=channels * 4,n_heads=8,
                                              n_layers=Config.num_layers,pad_index=0,device=Config.device)
            self.projection = nn.Linear(channels, out_dim, bias=False)



    def encode_src(self, src):
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def decode_trg(self, trg, memory, mask=True):
        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        if not mask:
            out = self.decoder(tgt=trg, memory=memory) + trg_start
        else:
            trg_mask = gen_trg_mask(out_sequence_len, trg.device)
            out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def eval_decode_trg(self, trg,memory):
        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)
        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def eval_forward(self,src,target,forcast_step):
        if Config.eval_type == 'back_encoder':
            forcast_step = target.shape[1]
            output_list = []

            for step in range(forcast_step):
                cur_target = torch.unsqueeze(src[:,-1, :],dim=1)
                encoder_out = self.encode_src(src)
                output = self.decode_trg(trg=cur_target, memory=encoder_out, mask=True)
                output = output[:, 0, :]
                if output.dim() == 2:
                    output = torch.unsqueeze(output, dim=1)
                src = torch.cat((src[:,:-1,:],output),dim=1)
                # cur_target111
                output_list.append(output)
            result = torch.cat(output_list, 1)
            return  result



        encoder_out = self.encode_src(src)

        if Config.eval_type == 'a':
            cur_target = torch.unsqueeze(src[:,-1, :],dim=1)
            # init_target = torch.unsqueeze(src[:,-1, :],dim=1)
            # output = None
            result = None
            # 自回归时间序列预测
            for step in range(forcast_step):
                cur_target = F.pad(cur_target, (0, 0, 0, forcast_step - step - 1))
                output = self.decode_trg(trg=cur_target, memory=encoder_out, mask=True)
                output = output[:,step,:]
                if output.dim() == 2:
                    output = torch.unsqueeze(output,dim=1)
                result = torch.cat((result,output),1) if not result is None else output
                cur_target = torch.cat((cur_target[:,:step+1,:], output), 1)
            return result
        elif Config.eval_type == 'half_in':
            output_list = []
            half = int(target.shape[1] / 2)
            target = target[:,:half,:]

            cur_target = target
            # 预测次数是
            for step in range(half + 1):
                cur_target = F.pad(cur_target, (0, 0, 0, forcast_step - step ))
                output = self.decode_trg(trg=cur_target, memory=encoder_out, mask=True)
                output = output[:,half-1 + step,:]
                output = torch.unsqueeze(output,dim=1)
                output_list.append(output)
                cur_target = torch.cat((cur_target[:,:half+step,:], output), 1)
            result = torch.cat(output_list, 1)
            return result
        elif Config.eval_type == 'truth_regression':
            output_list = []
            forcast_step = target.shape[1]
            for step in range(forcast_step):
                cur_target = F.pad(target[:,:step+1,:],(0,0,0,forcast_step - step - 1))
                print(f"输入数据 {target[:,:step+1,:].shape}")
                output = self.decode_trg(trg=cur_target, memory=encoder_out, mask=True)[:,step,:]
                print(f"输出，取出数据数据第{step}个数据")
                output = torch.unsqueeze(output,dim=1)
                output_list.append(output)
            result = torch.cat(output_list,1)
            return result
        elif Config.eval_type == 'test':
            cur_target = F.pad(target[:, :4, :], (0, 0, 0, 4))
            output = self.decode_trg(trg=cur_target, memory=encoder_out, mask=True)[:, 3, :]
            output = torch.unsqueeze(output, dim=1)
            return output


        else:
            return None



    def forward(self, x):
        src, trg = x

        if Config.swap_dimen:
            src = src.transpose(1,2)
            trg = trg.transpose(1,2)

        if Config.transformer_of_torch:
            src = self.encode_src(src)
            out = self.decode_trg(trg=trg, memory=src,mask=Config.mask_on_decoder)
        else:
            enc_out,_ = self.encoder(src)
            dec_out,_,_ = self.decoder(trg,src,enc_out)
            out = self.projection(dec_out)

        if Config.swap_dimen:
            out = out.transpose(1,2)
        return out