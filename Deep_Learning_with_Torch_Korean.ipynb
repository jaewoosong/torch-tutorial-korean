{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 토치를 사용한 딥 러닝: 60분에 끝내기"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "### 이 강좌의 목적\n",
    "* 토치와 신경망 패키지를 고수준에서 이해하기\n",
    "* CPU와 GPU에서 작은 신경망을 학습시켜 보기"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 토치란?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "토치는 루아[JIT]과 강력한 CPU 및 쿠다(CUDA) 뒷단에 기반한 과학 계산 프레임워크입니다.\n",
    "\n",
    "토치의 강점:\n",
    "\n",
    "* 효율적인 텐서 라이브러리 (NumPy와 같은) 및 효율적인 쿠다(CUDA) 뒷단\n",
    "* 신경망 패키지 -- 자동 미분이 지원되는 임의의 비순환 계산 그래프 만들기\n",
    "   * 빠른 CUDA와 CPU 뒷단 지원   \n",
    "* 양질의 커뮤니티와 기업 지원 - 커뮤니티에서 만들고 관리하는 수백개의 패키지\n",
    "* 다중 GPU와 신경망 병렬화 작업 용이\n",
    "\n",
    "http://torch.ch  \n",
    "https://github.com/torch/torch7/wiki/Cheatsheet"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 시작하기에 앞서"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 루아 기반이고, 속도가 빠른 루아-JIT(Just-in-time 컴파일러)에서 동작합니다.\n",
    "* 루아는 자바스크립트와 매우 비슷합니다.\n",
    "   * 변수들은 `local` 키워드를 쓰기 전에는 기본적으로 전역 변수입니다.\n",
    "* 테이블(`{}`)이라는 단 하나의 자료형만 가지고 있습니다. 해쉬 테이블과 배열로 사용 가능합니다.\n",
    "* 숫자를 1부터 셉니다.\n",
    "* `foo:bar()`는 `foo.bar(foo)`와 같습니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 시작합시다\n",
    "#### 문자열, 숫자, 테이블 - 간단한 설명"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "a = '안녕하세요'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "안녕하세요\t\n"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "b = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "b[1] = a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{\n",
       "  1 : 안녕하세요\n",
       "}\n"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "b[2] = 30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "안녕하세요\t\n",
       "30\t\n"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i=1,#b do -- 루아의 # 연산자는 길이를 뜻합니다.\n",
    "    print(b[i]) \n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 텐서"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "a = torch.Tensor(5,3) -- 초기화되지 않은 5x3 행렬을 만듭니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.9557  0.1468  0.4939\n",
       " 0.1668  0.9644  0.7250\n",
       " 0.5418  0.2848  0.2330\n",
       " 0.4625  0.7692  0.4746\n",
       " 0.0354  0.7377  0.6180\n",
       "[torch.DoubleTensor of size 5x3]\n",
       "\n"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = torch.rand(5,3)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "b=torch.rand(3,4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.5693  0.6139  1.4226  0.5961\n",
       " 1.3499  1.1109  1.1699  0.2167\n",
       " 0.4964  0.4059  0.8317  0.3436\n",
       " 1.0822  0.8386  1.1493  0.3481\n",
       " 1.0488  0.8925  0.8666  0.1202\n",
       "[torch.DoubleTensor of size 5x4]\n",
       "\n"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 행렬곱: 제 1 방식\n",
    "a*b "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.5693  0.6139  1.4226  0.5961\n",
       " 1.3499  1.1109  1.1699  0.2167\n",
       " 0.4964  0.4059  0.8317  0.3436\n",
       " 1.0822  0.8386  1.1493  0.3481\n",
       " 1.0488  0.8925  0.8666  0.1202\n",
       "[torch.DoubleTensor of size 5x4]\n",
       "\n"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 행렬곱: 제 2 방식\n",
    "torch.mm(a,b) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.5693  0.6139  1.4226  0.5961\n",
       " 1.3499  1.1109  1.1699  0.2167\n",
       " 0.4964  0.4059  0.8317  0.3436\n",
       " 1.0822  0.8386  1.1493  0.3481\n",
       " 1.0488  0.8925  0.8666  0.1202\n",
       "[torch.DoubleTensor of size 5x4]\n",
       "\n"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 행렬곱: 제 3 방식\n",
    "c=torch.Tensor(5,4)\n",
    "c:mm(a,b) -- a*b의 결과를 c에 저장합니다.\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 쿠다(CUDA) 텐서\n",
    "`:cuda` 함수를 사용해서 텐서를 GPU로 옮길 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.5693  0.6139  1.4226  0.5961\n",
       " 1.3499  1.1109  1.1699  0.2167\n",
       " 0.4964  0.4059  0.8317  0.3436\n",
       " 1.0822  0.8386  1.1493  0.3481\n",
       " 1.0488  0.8925  0.8666  0.1202\n",
       "[torch.CudaTensor of size 5x4]\n",
       "\n"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "require 'cutorch';\n",
    "a = a:cuda()\n",
    "b = b:cuda()\n",
    "c = c:cuda()\n",
    "c:mm(a,b) -- 이제 계산이 GPU에서 이루어집니다.\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 연습: 두 텐서 더하기\n",
    "https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchaddres-tensor1-tensor2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "function addTensors(a,b)\n",
    "    return a -- 이 부분을 고쳐주세요.\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 1  1\n",
       " 1  1\n",
       " 1  1\n",
       " 1  1\n",
       " 1  1\n",
       "[torch.DoubleTensor of size 5x2]\n",
       "\n"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = torch.ones(5,2)\n",
    "b = torch.Tensor(2,5):fill(4)\n",
    "print(addTensors(a,b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 신경망\n",
    "토치에서는 `nn` 패키지를 사용해서 신경망을 만들 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "require 'nn';"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "모듈(`Modules`)은 신경망에 있어 건물을 지을 때의 벽돌과 같은 존재입니다. 각 모듈은 그 자체로 신경망이기도 하지만, `containers` 명령어를 통해 다른 신경망과 연결되어서 더 복잡한 신경망을 만들 수도 있습니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "한 예로, 숫자를 분류해내는 다음 신경망을 보도록 하겠습니다.\n",
    "\n",
    "(파일 원본 출처: http://fastml.com/images/cifar/lenet5.png)\n",
    "![LeNet](data/deep_learning_with_torch/lenet5.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "단순한 피드 포워드 신경망입니다.\n",
    "입력값을 여러 계산에 단계별로 통과시켜서 최종 결과를 출력합니다.\n",
    "\n",
    "이렇게 입력값을 여러 계산 단계에 적용하는 신경망 컨테이너를 `nn.Sequential` 이라고 합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lenet5\n",
       "nn.Sequential {\n",
       "  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> output]\n",
       "  (1): nn.SpatialConvolution(1 -> 6, 5x5)\n",
       "  (2): nn.ReLU\n",
       "  (3): nn.SpatialMaxPooling(2x2, 2,2)\n",
       "  (4): nn.SpatialConvolution(6 -> 16, 5x5)\n",
       "  (5): nn.ReLU\n",
       "  (6): nn.SpatialMaxPooling(2x2, 2,2)\n",
       "  (7): nn.View(400)\n",
       "  (8): nn.Linear(400 -> 120)\n",
       "  (9): nn.ReLU\n",
       "  (10): nn.Linear(120 -> 84)\n",
       "  (11): nn.ReLU\n",
       "  (12): nn.Linear(84 -> 10)\n",
       "  (13): nn.LogSoftMax\n",
       "}\t\n"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "net = nn.Sequential()\n",
    "net:add(nn.SpatialConvolution(1,6,5,5))  -- 입력 그림 채널 1, 출력 채널 6, 합성곱 커널 5x5\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.SpatialMaxPooling(2,2,2,2))   -- 2x2의 범위에서 최대값을 찾는 맥스 풀링 계산\n",
    "net:add(nn.SpatialConvolution(6,16,5,5))\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.SpatialMaxPooling(2,2,2,2))\n",
    "net:add(nn.View(16*5*5))                 -- 16x5x5의 3차원 텐서를 16x5x5의 1차원 텐서로 변환\n",
    "net:add(nn.Linear(16*5*5, 120))          -- 모두 연결된(fully-connected) 레이어: 입력과 가중치간의 행렬곱입니다.\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.Linear(120, 84))\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.Linear(84, 10))               -- 10은 신경망의 출력 개수입니다. (이 경우 10개의 숫자)\n",
    "net:add(nn.LogSoftMax())                 -- 출력을 로그 확률로 변환합니다. 분류 문제를 풀 때 유용합니다.\n",
    "\n",
    "print('Lenet5\\n' .. net:__tostring());"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "nn 컨테이너의 다른 예제는 다음 그림에서 볼 수 있습니다.\n",
    "\n",
    "(파일 원본 출처: https://raw.githubusercontent.com/soumith/ex/gh-pages/assets/nn_containers.png)\n",
    "![containers](data/deep_learning_with_torch/nn_containers.png)\n",
    "\n",
    "토치의 신경망 모듈은 각각 자동 미분 기능을 가지고 있습니다.\n",
    "각 모듈은 주어진 입력에 대한 출력을 계산하는 `:forward(input)` 함수를 통해서 입력값을 신경망 내부에서 다음 단계로 전달합니다.\n",
    "또한 각 모듈은 신경망 내의 신경세포(뉴런)를 입력받은 기울기에 따라 미분하는 `:backward(input, gradient)` 함수도 가지고 있습니다.\n",
    "이 계산은 미분의 연쇄법칙을 통해 이루어집니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "input = torch.rand(1,32,32) -- 무작위 텐서를 신경망에 입력합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "output = net:forward(input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-2.2677\n",
       "-2.2567\n",
       "-2.3230\n",
       "-2.2966\n",
       "-2.3515\n",
       "-2.3425\n",
       "-2.3171\n",
       "-2.2450\n",
       "-2.2870\n",
       "-2.3454\n",
       "[torch.DoubleTensor of size 10]\n",
       "\n"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "net:zeroGradParameters() -- 신경망의 내부 기울기 버퍼를 0으로 만듭니다. (나중에 자세히 설명할 것입니다.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "gradInput = net:backward(input, torch.rand(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "  1\n",
       " 32\n",
       " 32\n",
       "[torch.LongStorage of size 3]\n",
       "\n"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(#gradInput)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 기준: 손실 함수 정의하기\n",
    "특정 작업을 위한 모델이 필요하다면 그 모델이 일을 얼마나 잘 하고 있는지를 그 모델에게 알려줘야 합니다. 모델의 일처리 능력을 측정하는 함수를 __손실 함수__라고 합니다.\n",
    "\n",
    "일반적인 손실 함수는 모델의 출력값과 참값을 받아서 모델의 일처리 결과를 수치화합니다.\n",
    "\n",
    "그 후 모델은 스스로를 고쳐서 손실을 줄입니다.\n",
    "\n",
    "토치에서 손실 함수는 신경망 모듈과 똑같이 구현되어 있으며 자동 미분을 지원합니다.\n",
    "손실 함수는 두 함수를 가지고 있습니다 - `forward(input, target)`, `backward(input, target)`\n",
    "\n",
    "예를 들자면,"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "criterion = nn.ClassNLLCriterion() -- 다중 분류를 위해 음의 로그 가능도를 기준으로 잡았습니다.\n",
    "criterion:forward(output, 3) -- 참값이 3이었다고 합시다.\n",
    "gradients = criterion:backward(output, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "gradInput = net:backward(input, gradients)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### 지금까지 배운 것 복습\n",
    "* 신경망은 여러 계산 단계를 가질 수 있다.\n",
    "* 신경망은 입력을 받아서 `:forward`를 통해 출력값을 만든다.\n",
    "* 기준은 신경망의 손실과 출력값에 대한 기울기를 계산한다.\n",
    "* 신경망은 되먹임 과정(backward pass)에서 (입력값, 기울기) 쌍을 사용하고 신경망의 각 단계와 그 단계의 신경세포(뉴런)의 기울기도 계산한다.\n",
    "\n",
    "##### 더 자세한 내용\n",
    "> 신경망이 학습 가능한 인자들을 가지고 있는지의 여부.\n",
    "\n",
    "합성곱 레이어는 입력값과 주어진 문제를 풀기에 적합한 합성곱 커널을 학습합니다.\n",
    "최대값 풀링 레이어는 학습 가능한 인자를 가지고 있지 않습니다. 그저 범위 안의 최대값을 찾을 뿐입니다.\n",
    "\n",
    "토치에서 학습 가능한 가중치를 가지고 있는 레이어는 일반적으로 가중치(`.weight`) 항목을 가지고 있습니다. (추가로 편향값(`.bias`) 항목을 갖는 경우도 있습니다.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1,1,.,.) = \n",
       "  0.4043 -0.2091\n",
       " -0.4127  0.3769\n",
       "\n",
       "(2,1,.,.) = \n",
       " -0.4682  0.0381\n",
       " -0.4462 -0.1156\n",
       "\n",
       "(3,1,.,.) = \n",
       " -0.4267 -0.1290\n",
       "  0.1416  0.0612\n",
       "[torch.DoubleTensor of size 3x1x2x2]\n",
       "\n"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m = nn.SpatialConvolution(1,3,2,2) -- 3개의 2x2 커널을 학습합니다.\n",
    "print(m.weight) -- 처음에는 가중치가 무작위로 선택됩니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.2715\n",
       " 0.3211\n",
       " 0.0467\n",
       "[torch.DoubleTensor of size 3]\n",
       "\n"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(m.bias) -- 합성곱 레이어의 계산은 다음과 같습니다. 출력 = 합성곱(입력, 가중치) + 편향값"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "학습 가능한 레이어에는 중요한 항목이 두 개 더 있습니다. gradWeight와 gradBias 입니다.\n",
    "gradWeight는 레이어의 각 가중치에 대한 기울기를 누적하고, gradBias는 레이어의 각 편향값에 대한 기울기를 누적합니다.\n",
    "\n",
    "#### 신경망 학습시키기\n",
    "\n",
    "신경망은 스스로를 조정하기 위해 일반적으로 다음 작업을 합니다. (확률적 경사 하강법을 사용할 경우)\n",
    "> 가중치 = 가중치 + (학습률 * gradWeight) [식 1]\n",
    "\n",
    "시간이 흐름에 따라 이 식은 출력값의 손실을 줄이는 쪽으로 신경망의 가중치를 조정할 것입니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "자, 이제 한 가지 빠진 퍼즐 조각에 대해 이야기할 때가 되었습니다. 누가 여러분의 신경망의 각 레이어를 방문해서 각 가중치를 식 1에 따라 갱신할까요?\n",
    "\n",
    "여러 답이 있을 수 있지만, 우리는 가장 간단한 답을 사용하도록 하겠습니다.\n",
    "우리는 신경망 모듈에 포함되어 있는 간단한 SGD 학습기를 사용합니다. (링크: [__nn.StochasticGradient__](https://github.com/torch/nn/blob/master/doc/training.md#stochasticgradientmodule-criterion))\n",
    "\n",
    "이 학습기의 `:train(dataset)` 함수는 주어진 데이터셋을 받아서 단순히 그때그때 다른 샘플을 신경망에 제공하는 방식으로 여러분의 신경망을 학습시킵니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 데이터는요?\n",
    "\n",
    "일반적으로 여러분이 그림, 글자, 오디오나 비디오 데이터를 다룰 경우 기본 함수인 [__image.load__](https://github.com/torch/image#res-imageloadfilename-depth-tensortype)나 [__audio.load__](https://github.com/soumith/lua---audio#usage) 등을 사용해서 데이터를 토치 텐서나 루아 테이블에 여러분에게 편한대로 입력할 수 있습니다.\n",
    "\n",
    "간단한 데이터를 사용해서 우리의 신경망을 학습시켜 봅시다.\n",
    "\n",
    "우리는 '비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭'의 클래스를 가진 CIFAR-10 데이터셋을 사용할 것입니다.\n",
    "CIFAR-10의 그림 파일들은 3x32x32, 즉 32x32 픽셀 크기의 3채널 컬러 이미지입니다.\n",
    "\n",
    "(파일 원본 출처: https://raw.githubusercontent.com/soumith/ex/gh-pages/assets/cifar10.png)\n",
    "![CIFAR-10 그림 파일](data/deep_learning_with_torch/cifar10.png)\n",
    "\n",
    "이 데이터셋에는 총 50,000개의 학습용 이미지와 10,000개의 테스트용 이미지가 들어 있습니다.\n",
    "\n",
    "__우리의 첫 토치 신경망을 학습시키기까지는 다섯 단계가 남아 있습니다.__\n",
    "1. 데이터 불러오고 정규화하기\n",
    "2. 신경망 정의하기\n",
    "3. 손실 함수 정의하기\n",
    "4. 학습 데이터에 대해 신경망 학습시키기\n",
    "5. 테스트 데이터에 대해 신경망 테스트하기\n",
    "\n",
    "__1. 데이터 불러오고 정규화하기__\n",
    "\n",
    "시간을 아끼기 위해 우리는 미리 50000x3x32x32(학습용)와 10000x3x32x32(테스트용)의 4차원 토치 바이트텐서(`ByteTensor`)를 준비해 두었습니다.\n",
    "데이터를 불러오고 살펴봅시다.\n",
    "\n",
    "__(역자 주)__\n",
    "\n",
    "원 문서에는 학습용 데이터가 50000개라고 되어 있는데, 실제로 자료를 다운받아보면 학습용 데이터가 10000개입니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Archive:  cifar10torchsmall.zip\n",
       "  inflating: cifar10-test.t7         "
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "\n",
       "  inflating: cifar10-train.t7        "
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "\n"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "require 'paths'\n",
    "if (not paths.filep(\"cifar10torchsmall.zip\")) then\n",
    "    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')\n",
    "    os.execute('unzip cifar10torchsmall.zip')\n",
    "    -- 역자 주: 54MB 짜리 압축 파일을 다운받습니다.\n",
    "    -- 역자 주: 압축을 풀면 나오는 데이터 파일을 data/deep_learning_with_torch 디렉토리에 저장해 두었습니다.\n",
    "end\n",
    "trainset = torch.load('cifar10-train.t7')\n",
    "testset = torch.load('cifar10-test.t7')\n",
    "classes = {'비행기', '자동차', '새', '고양이', '사슴',\n",
    "           '개', '개구리', '말', '배', '트럭'}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{\n",
       "  data : ByteTensor - size: 10000x3x32x32\n",
       "  label : ByteTensor - size: 10000\n",
       "}\n"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(trainset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 10000\n",
       "     3\n",
       "    32\n",
       "    32\n",
       "[torch.LongStorage of size 4]\n",
       "\n"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(#trainset.data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "재미를 위해 이미지 하나를 출력해 보겠습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "Console does not support images"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 32,
       "width": 32
      }
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "자동차\t\n"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "itorch.image(trainset.data[100]) -- 데이터셋의 100번째 이미지 출력\n",
    "print(classes[trainset.label[100]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "이제 __nn.StochasticGradient__에 사용할 데이터셋을 준비하기 위해 [설명서](https://github.com/torch/nn/blob/master/doc/training.md#traindataset)에 따라 몇 가지 해야 할 일이 있습니다.\n",
    "1. 데이터셋은 `:size()` 함수를 구현하고 있어야 합니다.\n",
    "2. 데이터셋은 `dataset[i]`가 데이터셋의 `i`번째 샘플을 반환하도록 하는 `[i]` 인덱스 연산자를 구현하고 있어야 합니다.\n",
    "\n",
    "둘 다 금방 할 수 있습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "-- setmetatable은 지금은 무시합시다. 이 학습 자료의 범위를 벗어난 내용으로, 인덱스 연산자를 설정하는 기능을 합니다.\n",
    "setmetatable(trainset, \n",
    "    {__index = function(t, i) \n",
    "                    return {t.data[i], t.label[i]} \n",
    "                end}\n",
    ");\n",
    "trainset.data = trainset.data:double() -- 바이트텐서의 데이터를 더블텐서로 변환합니다.\n",
    "\n",
    "function trainset:size() \n",
    "    return self.data:size(1) \n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000\t\n"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(trainset:size()) -- 그냥 테스트 한 번 해보려고요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{\n",
       "  1 : DoubleTensor - size: 3x32x32\n",
       "  2 : 2\n",
       "}\n"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "Console does not support images"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 32,
       "width": 32
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(trainset[33]) -- 33번 샘플을 불러옵니다.\n",
    "itorch.image(trainset[33][1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__여러분이 데이터를 다루는 과정에서 할 수 있는 중요한 일 중 하나는 (데이터 과학이나 기계 학습 영역에서 일반적으로) 데이터 평균을 0.0으로, 표준편차를 1.0으로 맞추는 일입니다.__\n",
    "\n",
    "우리의 데이터 관리의 최종 단계로 그 작업을 해 봅시다.\n",
    "\n",
    "이 일을 하기 위해 여러분께 텐서 인덱싱 연산자를 소개해 드리겠습니다.\n",
    "다음 예제를 봐 주세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- 결과: {모든 이미지, 채널 1, 모든 수직 방향 픽셀, 모든 수평 방향 픽셀}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 10000\n",
       "     1\n",
       "    32\n",
       "    32\n",
       "[torch.LongStorage of size 4]\n",
       "\n"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(#redChannel)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "인덱싱 연산자는 ___[{ }]___에서부터 시작됩니다. ___{}___를 사용해서 해당 차원의 모든 원소를 뽑아올 수도 있고, 원소 인덱스가 ___i___일 때 ___{i}___를 사용해서 특정 원소를 선택할 수도 있습니다. ___{i1, i2}___를 사용해서 범위 내의 원소를 출력할 수도 있습니다. 예를 들어 ___{3,5}___는 3, 4, 5번째 원소를 출력합니다.\n",
    "\n",
    "__연습: 데이터에서 150번째부터 300번째까지의 원소를 선택해보세요.__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 151\n",
       "   3\n",
       "  32\n",
       "  32\n",
       "[torch.LongStorage of size 4]\n",
       "\n"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 해보세요!\n",
    "-- 역자 주: 저는 이렇게 풀었습니다.\n",
    "data150to300 = trainset.data[{ {150,300}, {}, {}, {} }]\n",
    "print(#data150to300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "데이터의 각 원소에서 평균값을 빼고 표준편차를 조정하는 내용으로 돌아갑시다. 위에서 배운 인덱싱 연산자를 사용하면 이 작업이 쉽습니다. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "채널 1, 평균: 125.83175029297\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "채널 1, 표준편차: 63.143400842609\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "채널 2, 평균: 123.26066621094\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "채널 2, 표준편차: 62.369209019002\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "채널 3, 평균: 114.03068681641\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "채널 3, 표준편차: 66.965808411114\t\n"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean = {} -- 평균값. 나중에 테스트셋을 정규화 할 때 사용합니다.\n",
    "stdv = {} -- 나중을 위해 표준편차를 저장합니다.\n",
    "for i=1,3 do -- 각각의 이미지 채널에 대해서\n",
    "    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- 평균값 추정\n",
    "    print('채널 ' .. i .. ', 평균: ' .. mean[i])\n",
    "    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])    -- 평균값을 빼기\n",
    "    \n",
    "    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()  -- 표준편차 추정\n",
    "    print('채널 ' .. i .. ', 표준편차: ' .. stdv[i])\n",
    "    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])     -- 표준편차 조정\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "눈치채셨다시피 이제 학습 데이터가 정규화되어 사용될 준비가 되었습니다.\n",
    "\n",
    "__2. 우리의 신경망을 정의할 때입니다.__\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**연습:** 위의 __신경망__ 항목에서 신경망을 복사해서 3채널 이미지를 입력받을 수 있도록 수정해보세요. (위의 신경망은 1채널 이미지만 입력받도록 되어 있습니다.)\n",
    "도움: 맨 처음 레이어에서만 숫자를 1에서 3으로 바꾸면 됩니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 여기에 연습해보세요!\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__해답:__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "net = nn.Sequential()\n",
    "net:add(nn.SpatialConvolution(3,6,5,5))  -- 입력 그림 채널 3, 출력 채널 6, 합성곱 커널 5x5\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.SpatialMaxPooling(2,2,2,2))   -- 2x2의 범위에서 최대값을 찾는 맥스 풀링 계산\n",
    "net:add(nn.SpatialConvolution(6,16,5,5))\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.SpatialMaxPooling(2,2,2,2))\n",
    "net:add(nn.View(16*5*5))                 -- 16x5x5의 3차원 텐서를 16x5x5의 1차원 텐서로 변환\n",
    "net:add(nn.Linear(16*5*5, 120))          -- 모두 연결된(fully-connected) 레이어: 입력과 가중치간의 행렬곱입니다.\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.Linear(120, 84))\n",
    "net:add(nn.ReLU())                       -- 비선형\n",
    "net:add(nn.Linear(84, 10))               -- 10은 신경망의 출력 개수입니다. (이 경우 10개의 숫자)\n",
    "net:add(nn.LogSoftMax())                 -- 출력을 로그 확률로 변환합니다. 분류 문제를 풀 때 유용합니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__3. 손실 함수를 정의합시다.__\n",
    "\n",
    "로그 가능도 분류 손실을 사용하도록 하겠습니다. 대부분의 분류 작업에 유용합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "criterion = nn.ClassNLLCriterion()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__4. 신경망 학습시키기.__\n",
    "\n",
    "여기부터 모든 게 재미있어집니다!\n",
    "먼저 __nn.StochasticGradient__ 객체를 정의합시다. 그 후에 우리의 데이터셋을 이 객체의 ___:train___ 함수에 입력하면 그때부터 재미있는 일이 시작됩니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "trainer = nn.StochasticGradient(net, criterion)\n",
    "trainer.learningRate = 0.001\n",
    "trainer.maxIteration = 5 -- 5세대에 걸쳐서만 학습을 시킵시다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "# StochasticGradient: training\t\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 2.2280689231792\t\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.8978251628878\t\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.6883114820741\t\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.5752597623032\t\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.4904176074658\t\n",
       "# StochasticGradient: you have reached the maximum number of iterations\t\n",
       "# training error = 1.4904176074658\t\n",
       "\n"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer:train(trainset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### (역자 주)\n",
    "i7-4790k에서 5세대를 학습하는 데에 45분이 걸렸습니다. 한 세대 당 9분 정도 걸리네요. 꽤 긴 시간입니다.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "__5. 신경망을 테스트하고 정확도 출력하기.__\n",
    "\n",
    "학습 데이터셋 전체를 5번 반복 사용해서 신경망을 학습시켰습니다.\n",
    "그렇지만 진짜로 신경망이 뭔가를 배웠는지를 확인해봐야 합니다.\n",
    "신경망이 클래스 라벨을 예측하는지를 보고, 예측값을 참값과 비교해봄을 통해서 신경망을 확인해 보겠습니다. 만약 예측이 정확하다면 그 샘플을 올바른 예측 리스트에 추가합니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "첫번째 단계입니다. 테스트 셋과 친해지기 위해 이미지 하나를 출력해 보겠습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "말\t\n"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "Console does not support images"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 32,
       "width": 32
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(classes[testset.label[100]])\n",
    "itorch.image(testset.data[100])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "이 작업이 끝났으니 이제 테스트 데이터를 학습 데이터의 평균과 표준편차를 사용해서 정규화합시다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "testset.data = testset.data:double() -- 바이트 텐서를 더블 텐서로 변환합니다.\n",
    "for i=1,3 do -- 각 채널에 대해서\n",
    "    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- 평균값을 뺍니다.\n",
    "    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])  -- 표준편차를 조정합니다.\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.59066009532189\t1.0665356205025\t\n"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 재미를 위해 100번째 예제의 평균과 표준편차를 출력해봅시다.\n",
    "horse = testset.data[100]\n",
    "print(horse:mean(), horse:std())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "자, 이제 우리의 신경망이 위 예제를 뭐라고 예측하는지 보도록 합시다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "말\t\n"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAI3UlEQVRIiU1WWa9lV3GuqjXts/fZZ7jDad/bd+iB7jZuB2xME5zugBFIeYgyvPDAU5Q/hPgHKIlAhgeElESNFCHhYBwnYIPdie20uxv3dKe+wxn3sIYqHu5tkqX1sEoq1VffV4MWPt450kYRIgACAABoa0HQN02MERFIIQAgoiLCM5ezg89tETk1kQgRCRERSRESae99Yk2IDKAAldXzgz1I3Cl71mpEAQAQCDEqVKR1iiLCp5j/P/rZOyVEFKJTbALRpDQhAQAREuHu3Y++/93vbW1v/vXf/f3S+Q0QQARRcLKz98k776yeX3/pz25pawUE4SxTEUnMf+SkiBBR6TMu2mhNSgGIIlSWfvnTn9y+/fMXVoev/Pmt5c11oy2wRIC8yD9+/ze/+Oed7+TZl772dUBCAAAEERFhkT8qDACESKSQABFIaTJaWWe01Rrx0ZO9qOmzx08ffvYotpG0Rqu0Vbu/v3ewu/Phbz/49du/aqsFAiAICJ/qQ4in9xSARZiTcBJh7TJrrVKkiChG3+0vd5QeMxwej1k4M7atZkeP7r/94x+/9+77Mx9mk0lb11lRAiCAAAALAojI/6n0vDaAANoYMkYBAEAi0t/89t++9dYvj0/Gd9751X+urYamufvRx+3+ExnvXhy6pvvCpStXXacIIREAo4CA8BmSyCkfQUREQCJi0hyDFwEibQymsNbP1wbF8bA3fvT49j/9SCHWvuIUR4V7+cLG9quv3fjqDbRZ8FEhCNFz4c9CAwALPJeOEVEv6mCtwna2+z+/++933x0/eeAkXr1weZiZVRM2l7vDoq+NJZMlUelk9vCdtzu97srWJUBKiQHgtMbCp82KiuC04Hg6EUfTmufjD37y5u03f3RwdFRuvHjUttc6MFAxU7Ay7JEynAIgeqG6ZTJ69fLlrde/tvHFV4ulldbHMwQQTgwAREQERGSMJiJ8cu/uf/3gH27/9F/ef7h/69rmbr4+SLOrXUqdkoEgBcOp63THaCBSpJwmie2kZbt97fN/8ZcbL12PieEsfWARTomU0kohojCrm+eW3/zHH/7i3v7VzbXro35lintPD0Sp3+3N/v3u/p29yYmPioyz2mqTWdvJ87zoSkyPdvZq7bZefKn2MYSQUkrMIca29d7Hxoem9T4m/fN//bc7+4temf3p9qrLVH14/HgRyzw5q6wzCaSOcDRvSg0c46JuyhAHZbeT59e3L269fqPbL7opni0CBE4CIiLALEiIRPqD+4+qxH91ZTsz2IRkmvko1w9OmlfW+q9vd+ogAtDPtHOkUQRgVjcsUGauJFUabY0RIkBQgAKsSECAhAUBSYlChVFdXRu+trk6C21Iyqe4XqhDn/ZqGVg6Pyj6RZZbLJwusqwoCucscgJm8l7aJl8eut5SStLGGAKHmEJkn1JIEhKHyGrVlG+8eL7n7MG8jokTYompb9KzhvfnCTgMMuUUcoqcglXQzZxxbur54bi5+3D3o0/unXhYHi1lnQyYBc6a9XTMFZG6ub12aVgw6sNZlQA8M3Hbp7hsuYpyUMVJUwl7AwkkzSJ/Ng8f7s4/3Dn69NmzJ8fjTx/uvf/ee4t6vra1xQC+9a1vm6YNPiBwCl5fWunUIaZU1a3XAuMYFETToaUOfd7Ixydxf0FG27LsHAW89+RkPFusltnlldL0hwswHYuF+HD3zq/fGo2uXYcmofiQklKqW+Tz+UwXRidULCkwUwqzOuUUNBmi1DV6ZKVGs1PF3c/G3lfKV6tlZ3OlsL1eUN1RmXfAA6eykxdHT9fThd7GJrEXToiSOL77yR3tYwStFYom1cvUccDdOkT2LSdQfDxtgp8FO5hH7jvcWD+XO1NhxpIR4P39Q+95qd+V44P7z04e7B3e/OpLqz3fVEEbcsa9ttLqGKM2klJ0GqxxGc7vz9OBNHYiiKSd63Z0BvPUSrVQh1atuJwj5VVbNdO3f/PbPM82Xlh1BjXRe/cfjuudV189N55URWkyl5XOqW9c2iTgwMAMQPT0aDqu4zTwsZemjbdu/snNr39l0fDB0bTxoa7qkFKS0+WGgSXLM0AMPmTWlb18bfuFaWogc/vj2d7J/OF4od64sJEkVZEjY5vCNERgUsgMSEqNq/T73XFedl5++cKlyxvDQddKIOKktI8RRATEECIkreTFiys3v3KhbWbDPJc2xeRnVa1j8F44sjCqxNAxtnZpJogpDHvu9dcufvnlrctby1Ubn+4eDr48Gg3KTx8/+9l/PLjzv8+atk0xOk1dp5Z7+Zeuja6N8q3BllLoRzlrVTVea61qLwJYe5kJ7fg0aRhisgBX1su/eePq+qg/my96BarNvHBFPy+/8Dlb1X539/jTx41B6ffzz22OvnXr2q0bF6MPzjrQFFKbWFYKpXdaWUScNIEZfGDjUx8CU2ROl86vdlwH0RqTat+mlkEBMudG33rlyuWt8zvHjffVuZXywtpKrywgetZWK6MUUAzLtptmh1ojKE4utkZja4AjswKFOrc0GLjpYjIoXWhaDs14NjuaLa7orV7Z6yp9vbfyxasWNLTtPHpxtivQOiPKlrE9juDyojdrJurGxnLlYxABhDawZ0FAQ7h5vuyd64zrqU5eQlBa1Qx7xyccaoscQ4iJZ/Pp+Gh3Xk0nzaSen1CMk5NdVAgCHIPNyJqOnjaxiRwTIAogaqWc4q7Vo41+VubWZUw67w+Utqs5k7FGhAF9uwjBk7NROCWpY9qbHK10Z12bT+fPesOVyHU7k8SBOCUA0YpEkAWMIqvVsN/Je12T567b190lzAq0JjNu1B92u13WpjtctWVPtMqHQzC2DtHb7MGsSqaoElRtY3v9w5OjRV1pFggJogg9/884Q8sbK42zx9NFp+i2DEEws1kTmppjAImQVGoBcBG842idy0qopvPBoJwgFrozrxYRmtaB90k3SZTSAEwADGgIux29trWUDwbjxZQlxNCezHHcNsKRpVUaEkNYtIowCLByoLWyEtJ8fjLRZh2EJEXvF8VSOT5e/AFKAUzeGKFaswAAAABJRU5ErkJggg==",
      "text/plain": [
       "Console does not support images"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 32,
       "width": 32
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(classes[testset.label[100]])\n",
    "itorch.image(testset.data[100])\n",
    "predicted = net:forward(testset.data[100])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " 0.0142\n",
       " 0.0060\n",
       " 0.0439\n",
       " 0.2261\n",
       " 0.0387\n",
       " 0.2259\n",
       " 0.0280\n",
       " 0.3698\n",
       " 0.0033\n",
       " 0.0441\n",
       "[torch.DoubleTensor of size 10]\n",
       "\n",
       "\n"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "-- 신경망의 출력값은 로그 확률입니다. 일반 확률로 변환하려면 e^x를 적용해야 합니다.\n",
    "print(predicted:exp())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "여러분은 신경망의 예측값을 볼 수 있습니다. 이미지가 주어지면 신경망은 각각의 항목에 대한 확률을 계산해 놓습니다.\n",
    "\n",
    "명확하게 하기 위해 각각의 확률을 항목으로 이름붙입시다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "비행기\t0.014225540403785\t\n",
       "자동차\t0.0060144075703004\t\n",
       "새\t0.043935607255332\t\n",
       "고양이\t0.22606220834357\t\n",
       "사슴\t0.038727290643215\t\n",
       "개\t0.22586520296665\t\n",
       "개구리\t0.027960620013114\t\n",
       "말\t0.3697799203826\t\n",
       "배\t0.0033444720941473\t\n",
       "트럭\t0.044084730327285\t\n"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i=1,predicted:size(1) do\n",
    "    print(classes[i], predicted[i])\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "좋아요, 좋습니다. 예제 하나의 결과는 좋지 않습니다만, 테스트 셋 전체에 대해서는 맞는 예측값이 몇 개나 있을까요?\n",
    "(역자 주: 말이라고 제대로 예측되었는데 왜 원문에서는 예제 예측 결과가 좋지 않다고 했는지 잘 모르겠습니다.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "correct = 0\n",
    "for i=1,10000 do\n",
    "    local groundtruth = testset.label[i]\n",
    "    local prediction = net:forward(testset.data[i])\n",
    "    local confidences, indices = torch.sort(prediction, true)  -- 인자의 true는 내림차순 정렬을 의미합니다.\n",
    "    if groundtruth == indices[1] then\n",
    "        correct = correct + 1\n",
    "    end\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4704\t47.04 % \t\n"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(correct, 100*correct/10000 .. ' % ')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "찍는 것보다는 훠어어얼씬 좋아 보입니다. 찍으면 (항목이 10개니까) 확률은 10%일 것입니다. 신경망이 뭔가를 배운 것 같아 보이네요.\n",
    "\n",
    "음, 신경망이 잘 동작한 항목과 잘 동작하지 않은 항목은 각각 무엇일까요?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}\n",
    "for i=1,10000 do\n",
    "    local groundtruth = testset.label[i]\n",
    "    local prediction = net:forward(testset.data[i])\n",
    "    local confidences, indices = torch.sort(prediction, true)  -- 인자의 true는 내림차순 정렬을 의미합니다.\n",
    "    if groundtruth == indices[1] then\n",
    "        class_performance[groundtruth] = class_performance[groundtruth] + 1\n",
    "    end\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### (역자 주)\n",
    "이 과정도 시간이 은근 (몇 분 정도) 걸립니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "비행기\t46.4 %\t\n",
       "자동차\t66 %\t\n",
       "새\t37.9 %\t\n",
       "고양이\t26.5 %\t\n",
       "사슴\t20.8 %\t\n",
       "개\t49.9 %\t\n",
       "개구리\t51.3 %\t\n",
       "말\t55.2 %\t\n",
       "배\t63.8 %\t\n",
       "트럭\t52.6 %\t\n"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for i=1,#classes do\n",
    "    print(classes[i], 100*class_performance[i]/1000 .. ' %')\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "좋아요, 이제는 뭘 할까요? 이 신경망을 GPU에서 돌리려면 어떻게 해야 할까요?\n",
    "\n",
    "#### cunn: 신경망을 쿠다를 사용해서 GPU에서 돌리기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "require 'cunn';"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "간단합니다. 신경망을 가져다가 GPU로 보냅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "net = net:cuda()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "그리고 기준(손실 함수)도 GPU로 보냅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "criterion = criterion:cuda()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "네, 이제는 데이터도 보냅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "trainset.data = trainset.data:cuda()\n",
    "trainset.label = trainset.label:cuda()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "좋습니다. 이제 GPU에서 학습시켜 보겠습니다. ^^ #엄청_간단함"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "trainer = nn.StochasticGradient(net, criterion)\n",
    "trainer.learningRate = 0.001\n",
    "trainer.maxIteration = 5 -- 다섯 세대만 학습시킵니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "# StochasticGradient: training\t\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.4157297985792\t\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.3435925675392\t\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.2754184183955\t\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.2101447115064\t\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "# current error = 1.1439341126084\t\n",
       "# StochasticGradient: you have reached the maximum number of iterations\t\n",
       "# training error = 1.1439341126084\t\n",
       "\n"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer:train(trainset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "왜 CPU에 비해서 엄!청!난! 속도 향상이 없을까요?\n",
    "왜냐하면 우리의 신경망이 정말정말 작기 때문입니다.\n",
    "\n",
    "#### (역자 주)\n",
    "제가 보기엔 엄청난 속도 향상이 있는 것 같습니다. 학습이 30초 만에 끝났습니다. CPU에서 작업할 때 45분 걸렸던 것에 비해 엄청나게 빠릅니다.\n",
    "\n",
    "**연습:** 신경망의 크기를 늘려 보시고(`nn.SpatialConvolution(...)`의 첫 번째와 두 번째 인자) GPU를 사용하면 얼마만큼의 속도 향상이 있는지 확인해보세요."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**배운 것**\n",
    "* 토치와 신경망 패키지를 고수준에서 이해한다.\n",
    "* 작은 신경망을 CPU와 GPU에서 학습시킨다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 이제는 뭘 할까요?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 엄청 멋진 신경망 그래프 그리기: https://github.com/torch/nngraph\n",
    "* 여러 GPU를 사용해서 이미지넷 데이터에 대해 학습시키기: https://github.com/soumith/imagenet-multiGPU.torch\n",
    "* 롱 쇼트 텀 메모리(LSTM)를 사용한 반복(recurrent) 신경망을 텍스트에 대해 학습시키기: https://github.com/wojzaremba/lstm\n",
    "* 더 많은 예제와 교재: https://github.com/torch/torch7/wiki/Cheatsheet\n",
    "* 토치 개발자들과 이야기하기: http://gitter.im/torch/torch7\n",
    "* 도움을 요청하기: http://groups.google.com/forum/#!forum/torch7\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "iTorch",
   "language": "lua",
   "name": "itorch"
  },
  "language_info": {
   "name": "lua",
   "version": "5.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
