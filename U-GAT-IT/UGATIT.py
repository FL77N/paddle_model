import numpy as np
import itertools
from glob import glob

import paddle
import paddle.fluid as fluid
from hapi.datasets.folder import ImageFolder
from hapi.vision.transforms import transforms
from paddle.fluid.layers import ones_like, zeros_like
from paddle.fluid.dygraph import L1Loss, BCELoss, MSELoss

from networks import *
from utils import *

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.resume = args.resume
        self.use_gpu = args.use_gpu
        self.result_dir = args.result_dir
        self.model_path = args.model_path

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
                # DataLoader #
        place = fluid.CUDAPlace(0)
        with fluid.dygraph.guard(place):
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size + 30)),
                transforms.RandomResizedCrop((self.img_size, self.img_size)),
                transforms.Permute(to_rgb=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Permute(to_rgb=True),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

            self.trainA = ImageFolder(os.path.join(self.dataset, 'trainA'), transform=train_transform)
            self.trainB = ImageFolder(os.path.join(self.dataset, 'trainB'), transform=train_transform)
            self.testA = ImageFolder(os.path.join(self.dataset, 'testA'), transform=test_transform)
            self.testB = ImageFolder(os.path.join(self.dataset, 'testB'), transform=test_transform)

            self.trainA = np.array(self.trainA)
            self.trainB = np.array(self.trainB)
            self.testA = np.array(self.testA)
            self.testB = np.array(self.testB)

            self.trainA = np.reshape(self.trainA,(len(self.trainA),  3 ,self.img_size,  self.img_size))
            self.trainB = np.reshape(self.trainB,(len(self.trainB),  3 ,self.img_size,  self.img_size))
            self.testA = np.reshape(self.testA,(len(self.testA),  3 ,self.img_size,  self.img_size))
            self.testB = np.reshape(self.testB,(len(self.testB), 3 ,self.img_size,  self.img_size))

            self.trainA_loader = paddle.batch(paddle.reader.shuffle(read_img(self.trainA), 3000), batch_size=self.batch_size)
            self.trainB_loader = paddle.batch(paddle.reader.shuffle(read_img(self.trainB), 3000), batch_size=self.batch_size)
            self.testA_loader = paddle.batch(paddle.reader.shuffle(read_img(self.testA), 100), batch_size=1)
            self.testB_loader = paddle.batch(paddle.reader.shuffle(read_img(self.testA), 100), batch_size=1)

            """ Define Generator, Discriminator """
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
            self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
            self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

            """ Define Loss """
            self.L1_loss = L1Loss()
            self.MSE_loss = MSELoss()
            self.BCE_loss = BCEWithLogitsLoss()

            """ Trainer """        
            self.G_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=(self.genA2B.parameters()+self.genB2A.parameters()), regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))
            self.D_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.999, parameter_list=(self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters()), regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))

    def train(self):

        place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()

        with fluid.dygraph.guard(place):

            self.genA2B.train() 
            self.genB2A.train()
            self.disGA.train()
            self.disGB.train()
            self.disLA.train()
            self.disLB.train()

            if self.resume:

                files_list = os.listdir(self.model_path)

                if len(files_list) > 0:

                    files = []
                    print("exist files")

                    for i in files_list:

                        file_ = os.path.splitext(i)[1]
                        files.append(file_)

                    if ".pdparams" in files_list or ".pdopt" in files_list:

                        print("exist model")
                        genA2B_para = fluid.load_dygraph(self.model_path+'g_A2B')
                        genB2A_para = fluid.load_dygraph(self.model_path+'g_B2A')
                        disGA_para = fluid.load_dygraph(self.model_path+'d_GA')
                        disGB_para = fluid.load_dygraph(self.model_path+'d_GB')
                        disLA_para = fluid.load_dygraph(self.model_path+'d_LA')
                        disLB_para = fluid.load_dygraph(self.model_path+'d_LB')
                        G_opt = fluid.load_dygraph(self.model_path+'G_op')
                        D_opt = fluid.load_dygraph(self.model_path+'D_op')

                        self.genA2B.load_dict(genA2B_para)
                        self.genB2A.load_dict(genB2A_para)
                        self.disGA.load_dict(disGA_para)
                        self.disGB.load_dict(disGB_para)
                        self.disLA.load_dict(disLA_para)
                        self.disLB.load_dict(disLB_para)
                        self.G_optim.set_dict(G_opt)
                        self.D_optim.set_dict(D_opt)
                
                        print(" [*] Load SUCCESS")
                    
                    else:

                        print(" No Model!")
                
                else:

                    print("No Files")
        
            # training loop
            print('training start !')
            start_iter = 1
            for step in range(start_iter, self.iteration + 1):

                trainA_iter = iter(self.trainA_loader())
                real_A = next(trainA_iter)
                real_A = paddle.fluid.dygraph.to_variable(np.array(real_A))
                real_A =  real_A / 255.0

                trainB_iter = iter(self.trainB_loader())
                real_B = next(trainB_iter)
                real_B = paddle.fluid.dygraph.to_variable(np.array(real_B))
                real_B =  real_B / 255.0

                # Update D
                self.D_optim.clear_gradients()

                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.dygraph.to_variable(ones_like(real_GA_logit))) + self.MSE_loss(fake_GA_logit, fluid.dygraph.to_variable(zeros_like(fake_GA_logit)))               
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.dygraph.to_variable(ones_like(real_GA_cam_logit))) + self.MSE_loss(fake_GA_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_GA_cam_logit)))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.dygraph.to_variable(ones_like(real_LA_logit))) + self.MSE_loss(fake_LA_logit, fluid.dygraph.to_variable(zeros_like(fake_LA_logit)))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.dygraph.to_variable(ones_like(real_LA_cam_logit))) + self.MSE_loss(fake_LA_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_LA_cam_logit)))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.dygraph.to_variable(ones_like(real_GB_logit))) + self.MSE_loss(fake_GB_logit, fluid.dygraph.to_variable(zeros_like(fake_GB_logit)))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.dygraph.to_variable(ones_like(real_GB_cam_logit))) + self.MSE_loss(fake_GB_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_GB_cam_logit)))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit,fluid.dygraph.to_variable(ones_like(real_LB_logit))) + self.MSE_loss(fake_LB_logit, fluid.dygraph.to_variable(zeros_like(fake_LB_logit)))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.dygraph.to_variable(ones_like(real_LB_cam_logit))) + self.MSE_loss(fake_LB_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_LB_cam_logit)))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_optim.minimize(Discriminator_loss)

                # Update G
                self.G_optim.clear_gradients()

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.dygraph.to_variable(ones_like(fake_GA_logit)))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.dygraph.to_variable(ones_like(fake_GA_cam_logit)))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.dygraph.to_variable(ones_like(fake_LA_logit)))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.dygraph.to_variable(ones_like(fake_LA_cam_logit)))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.dygraph.to_variable(ones_like(fake_GB_logit)))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.dygraph.to_variable(ones_like(fake_GB_cam_logit)))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.dygraph.to_variable(ones_like(fake_LB_logit)))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.dygraph.to_variable(ones_like(fake_LB_cam_logit)))

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, fluid.dygraph.to_variable(ones_like(fake_B2A_cam_logit))) + self.BCE_loss(fake_A2A_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_A2A_cam_logit)))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, fluid.dygraph.to_variable(ones_like(fake_A2B_cam_logit))) + self.BCE_loss(fake_B2B_cam_logit, fluid.dygraph.to_variable(zeros_like(fake_B2B_cam_logit)))

                G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.minimize(Generator_loss)
 
                # clip parameter of AdaILN and ILN, applied after optimizer step
                clip_rho(self.genA2B, vmin=0, vmax=1)
                clip_rho(self.genB2A, vmin=0, vmax=1)

                if step % 50 == 0:
                    
                    print("[%5d/%5d] d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, Discriminator_loss, Generator_loss))

                if step % self.print_freq == 0:

                    print("print img!")
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

                    for _ in range(train_sample_num):

                        trainA_iter = iter(self.trainA_loader())
                        real_A = next(trainA_iter)
                        real_A = paddle.fluid.dygraph.to_variable(np.array(real_A))
                        real_A =  real_A / 255.0

                        trainB_iter = iter(self.trainB_loader())
                        real_B = next(trainB_iter)
                        real_B = paddle.fluid.dygraph.to_variable(np.array(real_B))
                        real_B =  real_B / 255.0

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    for _ in range(test_sample_num):

                        testA_iter = iter(self.testA_loader())
                        real_A = next(testA_iter)
                        real_A = paddle.fluid.dygraph.to_variable(np.array(real_A))
                        real_A =  real_A / 255.0

                        testB_iter = iter(self.testB_loader())
                        real_B = next(testB_iter)
                        real_B = paddle.fluid.dygraph.to_variable(np.array(real_B))
                        real_B =  real_B / 255.0

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    cv2.imwrite(os.path.join(self.result_dir, 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, 'B2A_%07d.png' % step), B2A * 255.0)

                if step % self.save_freq == 0:

                    fluid.save_dygraph(self.genA2B.state_dict(), self.model_path+'g_A2B')
                    fluid.save_dygraph(self.genB2A.state_dict(), self.model_path+'g_B2A')
                    fluid.save_dygraph(self.disGA.state_dict(), self.model_path+'d_GA')
                    fluid.save_dygraph(self.disGB.state_dict(), self.model_path+'d_GB')
                    fluid.save_dygraph(self.disLA.state_dict(), self.model_path+'d_LA')
                    fluid.save_dygraph(self.disLB.state_dict(), self.model_path+'d_LB')
                    fluid.save_dygraph(self.G_optim.state_dict(), self.model_path+'g_A2B')
                    fluid.save_dygraph(self.G_optim.state_dict(), self.model_path+'g_B2A')
                    fluid.save_dygraph(self.D_optim.state_dict(), self.model_path+'d_GA')
                    fluid.save_dygraph(self.D_optim.state_dict(), self.model_path+'d_GB') 
                    fluid.save_dygraph(self.D_optim.state_dict(), self.model_path+'d_LA') 
                    fluid.save_dygraph(self.D_optim.state_dict(), self.model_path+'d_LB')     

    def test(self):


        place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            
            print("Test!!!")

            if self.resume:

                files_list = os.listdir(self.model_path)

                if len(files_list) > 0:

                    files = []
                    print("exist files")

                    for i in files_list:

                        file_ = os.path.splitext(i)[1]
                        files.append(file_)

                    if ".pdparams" and ".pdopt" in files:

                        print("exist model")
                        genA2B_para, G_opt = fluid.load_dygraph(self.model_path+'g_A2B')
                        genB2A_para, G_2 = fluid.load_dygraph(self.model_path+'g_B2A')
                        disGA_para, D_opt = fluid.load_dygraph(self.model_path+'d_GA')
                        disGB_para, D_1 = fluid.load_dygraph(self.model_path+'d_GB')
                        disLA_para, D_2 = fluid.load_dygraph(self.model_path+'d_LA')
                        disLB_para, D_3 = fluid.load_dygraph(self.model_path+'d_LB')

                        self.genA2B.set_dict(genA2B_para)
                        self.genB2A.set_dict(genB2A_para)
                        self.disGA.set_dict(disGA_para)
                        self.disGB.set_dict(disGB_para)
                        self.disLA.set_dict(disLA_para)
                        self.disLB.set_dict(disLB_para)
                        self.G_optim.set_dict(G_opt)
                        self.D_optim.set_dict(D_opt)
                    
                        print(" [*] Load SUCCESS")
                        
                    else:

                        print(" No Model!")
                    
                else:

                    print("No Files")

            self.genA2B.eval()
            self.genB2A.eval()

            for n in range(self.testA.shape[0]):
                
                testA_iter = iter(self.testA_loader())
                real_A = next(testA_iter)
                real_A = paddle.fluid.dygraph.to_variable(np.array(real_A))
                real_A =  real_A / 255.0

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

                if n%50 == 0:
                    
                    print("save result img!",n)
                    cv2.imwrite(os.path.join(self.result_dir, "test_img", 'A2B_%d.png' % (n + 1)), A2B * 255.0)

            for n in range(self.testB.shape[0]):

                testB_iter = iter(self.testB_loader())
                real_B = next(testB_iter)
                real_B = paddle.fluid.dygraph.to_variable(np.array(real_B))
                real_B =  real_B / 255.0

                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)
                if n%50 == 0:

                    print("save result img!",n)
                    cv2.imwrite(os.path.join(self.result_dir, "test_img", 'A2B_%d.png' % (n + 1)), B2A * 255.0)
            
    def GetTest_result(self):


        place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            
            print("Get results!!!")

            if self.resume:

                files_list = os.listdir(self.model_path)

                if len(files_list) > 0:

                    files = []
                    print("exist files")

                    for i in files_list:

                        file_ = os.path.splitext(i)[1]
                        files.append(file_)

                    if ".pdparams" and ".pdopt" in files:

                        print("exist model")
                        genA2B_para, G_opt = fluid.load_dygraph(self.model_path+'g_A2B')
                        genB2A_para, G_2 = fluid.load_dygraph(self.model_path+'g_B2A')
                        disGA_para, D_opt = fluid.load_dygraph(self.model_path+'d_GA')
                        disGB_para, D_1 = fluid.load_dygraph(self.model_path+'d_GB')
                        disLA_para, D_2 = fluid.load_dygraph(self.model_path+'d_LA')
                        disLB_para, D_3 = fluid.load_dygraph(self.model_path+'d_LB')

                        self.genA2B.set_dict(genA2B_para)
                        self.genB2A.set_dict(genB2A_para)
                        self.disGA.set_dict(disGA_para)
                        self.disGB.set_dict(disGB_para)
                        self.disLA.set_dict(disLA_para)
                        self.disLB.set_dict(disLB_para)
                        self.G_optim.set_dict(G_opt)
                        self.D_optim.set_dict(D_opt)
                    
                        print(" [*] Load SUCCESS")
                        
                    else:

                        print(" No Model!")
                    
                else:

                    print("No Files")

            self.genA2B.eval()
            self.genB2A.eval()

            for n in range(self.testA.shape[0]):
                
                testA_iter = iter(self.testA_loader())
                real_A = next(testA_iter)
                real_A = paddle.fluid.dygraph.to_variable(np.array(real_A))
                real_A =  real_A / 255.0

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

                if n%2:

                    print("save result img!",n)

                cv2.imwrite(os.path.join(self.result_dir, "test_results", 'A2B_%d.jpg' % (n + 1)), A2B * 255.0)

         