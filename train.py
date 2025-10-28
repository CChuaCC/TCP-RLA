import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
from augment.glcm import GlcmFeature
from torch.nn.functional import interpolate

from dataset.chinesepaintings import train_chinesepaintings, val_chinesepaintings, TransformFixMatch, x_u_split
from dataset.randaugment import RandAugmentMC
from models.wideresnet import build_wideresnet
from utils import AverageMeter, accuracy


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ["CUDA_VISIBLE_DEVICES"]="0"
logger = logging.getLogger(__name__)
best_acc = 0
img_size = 224

torch.cuda.empty_cache()

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

#啥子操作 没明白
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
   
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=14,
                        help='number of workers')
    parser.add_argument('--dataset', default='chinesepaintings', type=str,
                        choices=['chinesepaintings','cifar10', 'cifar100',],
                        help='dataset name')#选择哪个数据集
    parser.add_argument('--num-labeled', type=int, default=6,
                        help='number of labeled data')#多少张带标签的图片
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")#数据扩展来适应每个step的数据产生
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext','mobilevit'],
                        help='dataset name')#选择哪个模型
    parser.add_argument('--total-steps', default=6400, type=int,  #2**12
                        help='number of total steps to run')#这里使用总步数来进行训练（epoch可以由此计算出）
    parser.add_argument('--eval-step', default=100, type=int,
                        help='number of eval steps to run')#每多少个批次进行一次评估 
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()

    wandb.init(project='chinesePaintings100-128', config={"learning_rate": 0.0002, "batch_size": 16, "epoch":100})
    
    global best_acc
    #backbone的选择
    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'mobilevit':
            import models.mobilevit as models
            model = models.mobile_vit_xx_small(num_classes=args.num_classes)

       

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model
# gpu分布式训练——————有疑问需要补充学习
    # local_rank指进程号
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        # world_size(int, optional): 参与工作的进程数
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))
    # 随机种子————有疑问需要补充学习
    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    # 数据集选择
    if args.dataset == 'cifar10':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    
    elif args.dataset == 'chinesepaintings':
        args.num_classes = 6
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 50
            args.model_width = 64
        elif args.arch == 'mobilevit':
            args.model_depth = 6
            args.dim = 128
            args.num_classes = 6
            args.image_size= 64
            args.patch_size=4
            args.heads=8, 
            args.mlp_ratio=3
        
# 分布式
    # 进程号？读取数据按一个进程进行？？
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    transform_labeled = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=img_size,
                              padding=int(img_size * 0.125),
                              padding_mode='reflect'),
        #GlcmFeature(distances=[1], angles=[0]),
        # PIL image转换为张量
        transforms.ToTensor(),
        # mean均值 std标准差 数据转换为一维分布
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
       ])
    
    transform_val = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    
    base_dataset = train_chinesepaintings(imgs_path = 'tcp/data/TCP512/train/train.txt')
    j = 0
    num_labels = []
    for i in base_dataset:
        num_labels.append(int(base_dataset[j][1]))
        j += 1

    # print(num_labels)
        # 加载数据集——重点关注 https://blog.csdn.net/sinat_42239797/article/details/90641659
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, num_labels)  # np.asarray(base_dataset.data)[:, 1])#使用x_u_split函数将带标签与不带标签的数据索引分开：

    labeled_dataset = train_chinesepaintings(
       imgs_path = 'tcp/data/TCP512/train/train.txt', idx = train_labeled_idxs, transform = transform_labeled)

    unlabeled_dataset = train_chinesepaintings(
        imgs_path = 'tcp/data/TCP512/train/train.txt', idx = train_unlabeled_idxs, transform = TransformFixMatch(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))

    test_dataset = val_chinesepaintings(
        imgs_path = 'tcp/data/TCP512/val/val.txt', transform = transform_val)
    # 生成dataloader按一个进程进行？？？
    if args.local_rank == 0:
        torch.distributed.barrier()
    # 通过dataloader类，我们产生dataloader对象，dataset是抽象类不能实例化，通过dataloader实例化加载数据
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # dataloader （1）shuffle数据集打乱 （2）组成batch
    labeled_trainloader = DataLoader( 
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,  
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    # 分布式？为啥这又有分布式的代码
    # 模型调用又开了一个进程？？？
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # 调用模型
    model = create_model(args)
    # 分布式？
    if args.local_rank == 0:
        torch.distributed.barrier()
    # 模型加载到相应的设备中
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 优化器SGD 没太明白这些参数是在做什么
    # parameters指权重
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    # optimizer = torch.optim.Adam(grouped_parameters, betas=(0.9, 0.999), lr=args.lr)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    # 梯度清零
    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    # if args.amp:
    #     from apex import amp
    global best_acc
    test_accs = []
    end = time.time()
    # 总进程不止一个
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs): # epoch迭代 所有训练样本都参与了训练
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            # 分布式？？？？
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
            # batch 每次训练的样本是多少
        for batch_idx in range(args.eval_step):
            try:
                # 准备有标签数据，inputs输入数据 target指标签
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
                # print("_"*30)
                # print(inputs_x, targets_x)
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter) # 加载无标数据 弱+强
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)


            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            
        
            #把数据加载到设备
            inputs = interleave( 
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device) #torch.cat()是为了把多个tensor进行拼接而存在的
            #标签加载到设备
            targets_x = targets_x.to(args.device)
            #logits相当于模型预测的target（标签）
            
            logits = model(inputs)
           
            logits = de_interleave(logits, 2*args.mu+1)#转置处理，这一步的目的是将每个无标签样本的预测结果切分成 2*args.mu+1 个片段，每个片段是一个独立的预测结果，为后续计算准确度提供支持。
            logits_x = logits[:batch_size]#读入有标签数据
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2) #c将 logits 中从第 batch_size 个元素开始的后续元素切分为两份，分别为 logits_u_w 和 logits_u_s，得到的两份结果形状均为 [batch_size*args.mu, num_classes]。这一步的目的是将无标签数据的预测结果分成两份，一份用于计算伪标签，另一份用于计算无标签数据的损失。
            del logits #删除logits变量

            Lx = F.cross_entropy(logits_x, targets_x.long(), reduction='mean') #有标签数据函数交叉熵损失（cross_entrop），logits_x指什么？？？

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1) #通过弱增强数据送入softmax输出伪标签预测
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)#把预测的伪标签中的最大值赋值给无标签弱监督
            mask = max_probs.ge(args.threshold).float()#把预测的伪标签转换成float 方便模型训练 ，代码将一个阈值 args.threshold 应用于最大概率值，生成一个掩码 mask，该掩码将大于等于阈值的最大概率值设置为1，将小于阈值的最大概率值设置为0，并将掩码转换为 float 值。这个掩码 mask 将被用于在无标签数据的预测中选择高置信度的预测样本，以便将它们用于模型训练。

            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()#无标签数据的交叉熵损失 强增强预测的数据和无标签弱增强产生的伪标签进行交叉熵计算

            loss = Lx + args.lambda_u * Lu #交叉熵损失 有标签数据的损失通常比无标签数据的损失要重要得多，因为有标签数据是稀缺的，而无标签数据则很容易获得。因此，我们使用 args.lambda_u 参数来加权平衡两者的贡献，以确保模型在训练过程中充分利用有标签数据的信息。
            loss.backward()  # 反向传播 更新权重
            # if args.amp:
            #     # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     #     scaled_loss.backward()
            # else:
            #     loss.backward()  # 反向传播 更新权重

            losses.update(loss.item()) #loss本身是一个矩阵，加上.item（）输出的是数值
            losses_x.update(Lx.item()) #更新有标签数据损失
            losses_u.update(Lu.item()) #更新无标签数据损失
            optimizer.step()#更新权重
            scheduler.step()
            if args.use_ema:
                ema_model.update(model) #针对cost进行指数平滑
            model.zero_grad()#初始化权重

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())#产生的伪标签top5求均值然后更新？？
           
            if not args.no_progress:#实验中参数显示
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))#mask指伪标签的均值？？
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema #模型指数平滑
        else:
            test_model = model
        
        #记录实验数据
        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            wandb.log({'loss':losses.avg,'loss_x':losses_x.avg,'loss_u':losses_u.avg,'test_acc':test_acc,'epoch': epoch})
            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
    
    if args.local_rank in [-1, 0]:#如果进程结束，关闭该功能
        args.writer.close()
    wandb.finish()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
# '''对于每个未标记样本，FixMatch会将其输入到分类器中，得到其预测结果。然后将预测结果中概率前k大的标签对应的概率值相加，
# 得到一个置信度分数。如果这个分数超过了置信度阈值，则将该未标记样本视为可靠的，并将其添加到训练数据中。
# 如果这个分数低于置信度阈值，则将该未标记样本视为不可靠的，并将其丢弃。
# '''
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)#预测结果
            loss = F.cross_entropy(outputs, targets.long())#损失函数

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))#取前五预测值？？？为啥是5？
             
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    wandb.log({'top1.avg':top1.avg})
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
