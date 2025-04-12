import os
import json
import uuid
import time
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from collections import defaultdict

from PIL import Image
import requests
from loguru import logger

import plugins
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from plugins import *

import logging
import requests
import io
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import traceback
import copy
import threading
import urllib.parse

import random
import string
import hashlib
import re
from common.tmp_dir import TmpDir

@plugins.register(
    name="GeminiImage",
    desire_priority=20,
    hidden=False,
    desc="基于Google Gemini的图像生成插件",
    version="1.0.0",
    author="Lingyuzhou",
)
class GeminiImage(Plugin):
    """基于Google Gemini的图像生成插件
    
    功能：
    1. 生成图片：根据文本描述生成图片
    2. 编辑图片：根据文本描述修改已有图片
    3. 支持会话模式，可以连续对话修改图片
    4. 支持积分系统控制使用
    """
        
    # 注意：所有日志记录中，不要输出完整的base64编码数据，只记录长度或截取前20-100个字符
    # 完整base64数据会导致日志文件过大，特别是在处理多图的情况下
    
    # 请求体大小限制常量（单位：字节）- 限制为4MB，避免413错误
    MAX_REQUEST_SIZE = 4 * 1024 * 1024
    # 会话中保留的最大消息数量
    MAX_CONVERSATION_MESSAGES = 10
    
    # 会话类型常量
    SESSION_TYPE_GENERATE = "generate"  # 生成图片模式
    SESSION_TYPE_EDIT = "edit"          # 编辑图片模式
    SESSION_TYPE_REFERENCE = "reference" # 参考图编辑模式
    SESSION_TYPE_MERGE = "merge"        # 融图模式
    SESSION_TYPE_ANALYSIS = "analysis"   # 图片分析模式
    
    # 默认配置
    DEFAULT_CONFIG = {
        "enable": True,
        "gemini_api_key": "",
        "model": "gemini-2.0-flash-exp-image-generation",
        "commands": ["g生成图片", "g画图", "g画一个"],
        "edit_commands": ["g编辑图片", "g改图"],
        "reference_edit_commands": ["g参考图", "g编辑参考图"],
        "merge_commands": ["g融图"],
        "image_reverse_commands": ["g反推提示", "g反推"],
        "image_analysis_commands": ["g分析图片", "g识图"],
        "follow_up_commands": ["g追问"],
        "exit_commands": ["g结束对话", "g结束"],
        "print_model_commands": ["g打印对话模型", "g打印模型"],
        "switch_model_commands": ["g切换对话模型", "g切换模型"],
        "chat_commands": ["g对话"],
        "expand_commands": ["g扩写"],
        "enable_points": False,
        "generate_image_cost": 10,
        "edit_image_cost": 15,
        "save_path": "temp",
        "admins": [],
        "enable_proxy": False,
        "proxy_url": "",
        "use_proxy_service": True,
        "proxy_service_url": "",
        "translate_api_base": "https://open.bigmodel.cn/api/paas/v4",
        "translate_api_key": "",
        "translate_model": "glm-4-flash",
        "enable_translate": False,
        "translate_on_commands": ["g开启翻译", "g启用翻译"],
        "translate_off_commands": ["g关闭翻译", "g禁用翻译"],
        "reverse_prompt": ""
    }

    def __init__(self):
        """初始化插件配置"""
        try:
            super().__init__()
            
            # 载入配置
            self.config = super().load_config() or self._load_config_template()
            
            # 使用默认配置初始化
            for key, default_value in self.DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = default_value
            
            # 设置配置参数
            self.enable = self.config.get("enable", True)
            self.api_key = self.config.get("gemini_api_key", "")
            
            # 模型配置
            self.image_model = self.config.get("image_model", "gemini-2.0-flash-exp-image-generation")
            self.chat_model = self.config.get("chat_model", "gemini-2.0-flash-thinking-exp-01-21")
            # 可用模型列表
            self.chat_model_list = self.config.get("chat_model_list", [
                "gemini-2.0-flash-thinking-exp-01-21",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.5-pro-preview-03-25"
            ])

            # 获取baseurl配置
            self.base_url = self.config.get("base_url", "https://generativelanguage.googleapis.com")
            
            # 获取命令配置
            self.commands = self.config.get("commands", ["g生成图片", "g画图", "g画一个"])
            self.edit_commands = self.config.get("edit_commands", ["g编辑图片", "g改图"])
            self.reference_edit_commands = self.config.get("reference_edit_commands", ["g参考图", "g编辑参考图"])
            self.merge_commands = self.config.get("merge_commands", ["g融图"])
            self.image_reverse_commands = self.config.get("image_reverse_commands", ["g反推提示", "g反推"])
            self.image_analysis_commands = self.config.get("image_analysis_commands", ["g分析图片", "g识图"])
            self.follow_up_commands = self.config.get("follow_up_commands", ["g追问"])
            self.exit_commands = self.config.get("exit_commands", ["g结束对话", "g结束"])
            self.expand_commands = self.config.get("expand_commands", ["g扩写"])
            self.chat_commands = self.config.get("chat_commands", ["g对话", "g回答"])
            self.print_model_commands = self.config.get("print_model_commands", ["g打印对话模型", "g打印模型"])
            self.switch_model_commands = self.config.get("switch_model_commands", ["g切换对话模型", "g切换模型"])
            
            # 获取积分配置
            self.enable_points = self.config.get("enable_points", False)
            self.generate_cost = self.config.get("generate_image_cost", 10)
            self.edit_cost = self.config.get("edit_image_cost", 15)
            
            # 获取图片保存配置
            self.save_path = self.config.get("save_path", "temp")
            self.save_dir = os.path.join(os.path.dirname(__file__), self.save_path)
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 获取管理员列表
            self.admins = self.config.get("admins", [])
            
            # 获取代理配置
            self.enable_proxy = self.config.get("enable_proxy", False)
            self.proxy_url = self.config.get("proxy_url", "")
            
            # 获取代理服务配置
            self.use_proxy_service = self.config.get("use_proxy_service", True)
            self.proxy_service_url = self.config.get("proxy_service_url", "")
            
            # 获取翻译API配置
            self.enable_translate = self.config.get("enable_translate", True)
            self.translate_api_base = self.config.get("translate_api_base", "https://open.bigmodel.cn/api/paas/v4")
            self.translate_api_key = self.config.get("translate_api_key", "")
            self.translate_model = self.config.get("translate_model", "glm-4-flash")
            
            # 获取翻译控制命令配置
            self.translate_on_commands = self.config.get("translate_on_commands", ["g开启翻译", "g启用翻译"])
            self.translate_off_commands = self.config.get("translate_off_commands", ["g关闭翻译", "g禁用翻译"])
            
            # 获取提示词扩写配置
            self.expand_prompt = self.config.get("expand_prompt", "请帮我扩写以下提示词，使其更加详细和具体：{prompt}")
            self.expand_model = self.config.get("expand_model", "gemini-2.0-flash-thinking-exp-01-21")
            
            # 用户翻译设置缓存，用于存储每个用户的翻译设置
            self.user_translate_settings = {}  # 用户ID -> 是否启用翻译
            
            # 初始化会话状态，用于保存上下文
            self.conversations = defaultdict(list)  # 存储会话历史，默认初始化为空列表
            self.last_conversation_time = {}  # 记录每个会话的最后活动时间        
            self.conversation_session_types = {}  # 记录每个会话的类型
            self.conversation_expire_seconds = 180  # 会话过期时间(秒)，改为3分钟
            self.last_images = {}  # 记录每个会话最后生成的图片路径           

            self.waiting_for_reference_image = {}  # 用户ID -> 等待参考图片的提示词
            self.waiting_for_reference_image_time = {}  # 用户ID -> 开始等待参考图片的时间戳
            self.reference_image_wait_timeout = 180  # 等待参考图片的超时时间(秒)，3分钟
            
            # 初始化图片分析状态
            self.waiting_for_reverse_image = {}  # 用户ID -> 是否等待反推图片
            self.waiting_for_reverse_image_time = {}  # 用户ID -> 开始等待反推图片的时间戳
            self.reverse_image_wait_timeout = 180  # 等待反推图片的超时时间(秒)，3分钟
            
            # 初始化识图状态
            self.waiting_for_analysis_image = {}  # 用户ID -> 等待识图的问题
            self.waiting_for_analysis_image_time = {}  # 用户ID -> 开始等待识图的时间戳
            self.analysis_image_wait_timeout = 180  # 等待识图的超时时间(秒)，3分钟

            # 初始化融图状态
            self.waiting_for_merge_image = {}  # 用户ID -> 等待的融图提示词
            self.waiting_for_merge_image_time = {}  # 用户ID -> 开始等待融图的时间戳
            self.merge_image_wait_timeout = 180  # 等待融图图片的超时时间(秒)，3分钟
            self.merge_first_image = {}  # 用户ID -> 第一张图片数据            

            # 初始化图片缓存，用于存储用户上传的图片
            self.image_cache = {}  # 会话ID/用户ID -> {"data": 图片数据, "timestamp": 时间戳}
            self.image_cache_timeout = 600  # 图片缓存过期时间(秒)
            
            # 初始化追问状态
            self.last_analysis_image = {}  # 用户ID -> 最后一次识图的图片数据
            self.last_analysis_time = {}  # 用户ID -> 最后一次识图的时间戳
            self.follow_up_timeout = 180  # 追问超时时间(秒)，3分钟
            
            # 获取图片分析提示词
            self.reverse_prompt = self.config.get("reverse_prompt", "请详细分析这张图片的内容，包括主要对象、场景、风格、颜色等关键特征。如果图片包含文字，也请提取出来。请用简洁清晰的中文进行描述。")
            
            # 验证关键配置
            if not self.api_key:
                logger.warning("GeminiImage插件未配置API密钥")
            
            # 绑定事件处理函数
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
          
            logger.info("GeminiImage插件初始化成功")
            if self.enable_proxy:
                logger.info(f"GeminiImage插件已启用代理: {self.proxy_url}")
            
        except Exception as e:
            logger.error(f"GeminiImage插件初始化失败: {str(e)}")
            logger.exception(e)
            self.enable = False

    def on_handle_context(self, e_context: EventContext):
        """处理消息事件"""
        if not self.enable:
            return
        
        # 获取上下文
        context = e_context['context']
        
        # 清理过期会话和图片缓存
        self._cleanup_expired_conversations()
        self._cleanup_image_cache()
        
        # 处理图片消息 - 用于缓存用户发送的图片
        if context.type == ContextType.IMAGE:
            logger.info("接收到图片消息，开始处理")
            self._handle_image_message(e_context)
            return
            
        # 检查是否是文本消息
        if context.type != ContextType.TEXT:
            return
            
        # 获取消息内容
        content = context.content
        if not content:
            return
            
        # 获取用户ID
        context = e_context['context']
        user_id = context.kwargs.get("session_id")
        is_group = context.get("isgroup", False)
        
        # 获取消息对象
        msg = None
        if 'msg' in context.kwargs:
            msg = context.kwargs['msg']
            # 在群聊中，优先使用actual_user_id作为用户标识
            if is_group and hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                user_id = msg.actual_user_id
                logger.info(f"群聊中使用actual_user_id作为用户ID: {user_id}")
            elif not is_group:
                # 私聊中使用from_user_id
                if hasattr(msg, 'from_user_id') and msg.from_user_id:
                    user_id = msg.from_user_id
                    logger.info(f"私聊中使用from_user_id作为用户ID: {user_id}")
        
        if not user_id:
            logger.error("无法获取用户ID")
            return
            
        # 会话标识: 用户ID（不附加_generate后缀，保持一致性）
        conversation_key = user_id
        
        # 处理图片消息 - 用于缓存用户发送的图片
        if context.type == ContextType.IMAGE:
            self._handle_image_message(e_context)
            return
            
        # 处理文本消息
        if context.type != ContextType.TEXT:
            return
        
        content = context.content.strip()
        
        # 检查是否是打印模型命令
        for cmd in self.print_model_commands:
            if content == cmd:
                # 构建模型列表文本
                models_text = "Gemini可用对话模型：\n"
                for i, model in enumerate(self.chat_model_list, 1):
                    prefix = "👉" if model == self.chat_model else ""
                    models_text += f"{prefix}{i}. {model}\n"
                
                models_text += "\n如需切换请输入命令和模型序号，例如：g切换模型 3"
                reply = Reply(ReplyType.TEXT, models_text)
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是切换模型命令
        for cmd in self.switch_model_commands:
            if content.startswith(cmd):
                # 提取模型序号
                parts = content.split()
                if len(parts) < 2:
                    # 只输入了切换模型命令，没有指定模型序号
                    models_text = "Gemini可用对话模型：\n"
                    for i, model in enumerate(self.chat_model_list, 1):
                        prefix = "👉" if model == self.chat_model else ""
                        models_text += f"{prefix}{i}. {model}\n"
                    
                    models_text += "\n如需切换请输入命令和模型序号，例如：g切换模型 3"
                    reply = Reply(ReplyType.TEXT, models_text)
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                else:
                    # 尝试解析模型序号
                    try:
                        model_index = int(parts[1]) - 1  # 用户输入的是从1开始的序号
                        
                        if 0 <= model_index < len(self.chat_model_list):
                            # 有效的模型序号
                            new_model = self.chat_model_list[model_index]
                            self.chat_model = new_model
                            self.config["model"] = new_model
                            
                            # 更新配置文件
                            config_path = os.path.join(os.path.dirname(__file__), "config.json")
                            if os.path.exists(config_path):
                                with open(config_path, 'r', encoding='utf-8') as file:
                                    config_data = json.load(file)
                                    config_data["model"] = new_model
                                    with open(config_path, 'w', encoding='utf-8') as file:
                                        json.dump(config_data, file, ensure_ascii=False, indent=2)
                            
                            reply = Reply(ReplyType.TEXT, f"已切换对话模型: {new_model}")
                        else:
                            # 无效的模型序号
                            reply = Reply(ReplyType.TEXT, f"无效的模型序号：{model_index + 1}，可用序号范围：1-{len(self.chat_model_list)}")
                    except ValueError:
                        # 无法解析为整数
                        reply = Reply(ReplyType.TEXT, "请输入有效的模型序号，例如：g切换对话模型 3")
                
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是反推提示词命令
        for cmd in self.image_reverse_commands:
            if content == cmd:
                # 设置等待图片状态
                self.waiting_for_reverse_image[user_id] = True
                self.waiting_for_reverse_image_time[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请在3分钟内发送需要gemini反推提示词的图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
        # 检查是否是识图命令
        for cmd in self.image_analysis_commands:
            # 检查是否包含问题
            if content.startswith(cmd):
                question = content[len(cmd):].strip()
                # 设置等待图片状态，并保存问题
                self.waiting_for_analysis_image[user_id] = question if question else "分析这张图片的内容，包括主要对象、场景、风格、颜色等关键特征，用简洁清晰的中文进行描述。"
                self.waiting_for_analysis_image_time[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请在3分钟内发送需要gemini识别的图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
        # 检查是否是追问命令
        for cmd in self.follow_up_commands:
            if content.startswith(cmd):
                # 检查是否有最近的识图记录
                if user_id not in self.last_analysis_image or user_id not in self.last_analysis_time:
                    reply = Reply(ReplyType.TEXT, "没有找到最近的识图记录，请先使用识图功能")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查是否超时
                if time.time() - self.last_analysis_time[user_id] > self.follow_up_timeout:
                    # 清理状态
                    del self.last_analysis_image[user_id]
                    del self.last_analysis_time[user_id]
                    
                    reply = Reply(ReplyType.TEXT, "追问超时，请重新使用识图功能")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 提取追问问题
                question = content[len(cmd):].strip() if len(content) > len(cmd) else "请继续分析这张图片"
                # 添加中文回答要求
                question = question + "，请用简洁的中文进行回答。"
                
                try:
                    # 调用API分析图片
                    analysis_result = self._analyze_image(self.last_analysis_image[user_id], question)
                    if analysis_result:
                        # 更新时间戳
                        self.last_analysis_time[user_id] = time.time()
                        
                        # 添加追问提示
                        analysis_result += "\n💬3min内输入g追问+问题，可继续追问"
                        reply = Reply(ReplyType.TEXT, analysis_result)
                    else:
                        reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                except Exception as e:
                    logger.error(f"处理追问请求异常: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是提示词扩写命令
        for cmd in self.expand_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供需要扩写的提示词，格式：{cmd} [提示词]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                try:
                    # 发送处理中消息
                    processing_reply = Reply(ReplyType.TEXT, f"正在使用{self.expand_model}扩写提示词...")
                    e_context["channel"].send(processing_reply, e_context["context"])
                    
                    # 调用API进行提示词扩写
                    response = self._expand_prompt(prompt)
                    
                    if response:
                        # 添加用户提示到会话
                        user_message = {"role": "user", "parts": [{"text": prompt}]}
                        # 发送回复
                        reply = Reply(ReplyType.TEXT, response)
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                    else:
                        reply = Reply(ReplyType.TEXT, "提示词扩写失败，请稍后重试")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                except Exception as e:
                    logger.error(f"处理提示词扩写请求失败: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"处理提示词扩写请求失败: {str(e)}")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                return
                
        # 检查是否是对话命令
        for cmd in self.chat_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供对话内容，格式：{cmd} [内容]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                try:
                    # 发送处理中消息
                    processing_reply = Reply(ReplyType.TEXT, f"正在调用{self.chat_model}回答您的问题...")
                    e_context["channel"].send(processing_reply, e_context["context"])
                    
                    # 获取会话历史
                    conversation_history = self.conversations[conversation_key]
                    
                    # 翻译提示词
                    translated_prompt = self._translate_prompt(prompt, user_id)
                    
                    # 调用API进行对话
                    response = self._chat_with_gemini(translated_prompt, conversation_history)
                    
                    if response:
                        # 添加用户提示到会话
                        user_message = {"role": "user", "parts": [{"text": prompt}]}
                        conversation_history.append(user_message)
                        
                        # 添加助手回复到会话
                        assistant_message = {
                            "role": "model", 
                            "parts": [{"text": response}]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.last_conversation_time[conversation_key] = time.time()
                        
                        # 发送回复
                        reply = Reply(ReplyType.TEXT, response)
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                    else:
                        reply = Reply(ReplyType.TEXT, "对话失败，请稍后重试")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                except Exception as e:
                    logger.error(f"处理对话请求失败: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"处理对话请求失败: {str(e)}")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                return

        # 检查是否是翻译控制命令
        for cmd in self.translate_on_commands:
            if content == cmd:
                # 启用翻译
                self.user_translate_settings[user_id] = True
                reply = Reply(ReplyType.TEXT, "已开启前置翻译功能，接下来的图像生成和编辑将自动将中文提示词翻译成英文")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        for cmd in self.translate_off_commands:
            if content == cmd:
                # 禁用翻译
                self.user_translate_settings[user_id] = False
                reply = Reply(ReplyType.TEXT, "已关闭前置翻译功能，接下来的图像生成和编辑将直接使用原始中文提示词")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否在等待用户上传参考图片
        if user_id in self.waiting_for_reference_image:
            # 检查是否超时
            current_time = time.time()
            start_time = self.waiting_for_reference_image_time.get(user_id, 0)
            
            if current_time - start_time > self.reference_image_wait_timeout:
                # 超过3分钟，自动结束等待
                logger.info(f"用户 {user_id} 等待上传参考图片超时，自动结束流程")
                prompt = self.waiting_for_reference_image[user_id]
                
                # 清除等待状态
                del self.waiting_for_reference_image[user_id]
                if user_id in self.waiting_for_reference_image_time:
                    del self.waiting_for_reference_image_time[user_id]
                
                # 发送超时提示
                reply = Reply(ReplyType.TEXT, f"等待上传参考图片超时（超过{self.reference_image_wait_timeout//60}分钟），已自动取消操作。如需继续，请重新发送参考图编辑命令。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 获取之前保存的提示词
            prompt = self.waiting_for_reference_image[user_id]
            
            # 获取消息对象
            msg = None
            if 'msg' in context.kwargs:
                msg = context.kwargs['msg']
            
            # 先检查context.kwargs中是否有image_base64
            image_base64 = context.kwargs.get("image_base64")
            
            # 如果没有image_base64，使用统一的图片获取方法
            if not image_base64:
                # 使用统一的图片获取方法获取图片数据
                image_data = self._get_image_data(msg, "")  # 传入空字符串，让方法尝试从msg中获取图片
                
                # 如果获取到图片数据，转换为base64
                if image_data and len(image_data) > 1000:
                    try:
                        # 验证图片数据是否有效
                        Image.open(BytesIO(image_data))
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        logger.info(f"成功获取图片数据并转换为base64，大小: {len(image_data)} 字节")
                    except Exception as img_err:
                        logger.error(f"获取的图片数据无效: {img_err}")
            
            # 如果成功获取到图片数据
            if image_base64:
                # 清除等待状态
                del self.waiting_for_reference_image[user_id]
                if user_id in self.waiting_for_reference_image_time:
                    del self.waiting_for_reference_image_time[user_id]
                
                # 发送成功获取图片的提示
                success_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                e_context["reply"] = success_reply
                e_context.action = EventAction.BREAK_PASS
                e_context["channel"].send(success_reply, e_context["context"])
                
                # 处理参考图片编辑
                self._handle_reference_image_edit(e_context, user_id, prompt, image_base64)
                return
            else:
                # 用户没有上传图片，提醒用户
                reply = Reply(ReplyType.TEXT, "请上传一张图片作为参考图进行编辑。如果想取消操作，请发送\"g结束对话\"")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是结束对话命令
        if content in self.exit_commands:
            if conversation_key in self.conversations:
                # 清除会话数据
                del self.conversations[conversation_key]
                if conversation_key in self.last_conversation_time:
                    del self.last_conversation_time[conversation_key]
                if conversation_key in self.last_images:
                    del self.last_images[conversation_key]
                
                reply = Reply(ReplyType.TEXT, "已结束Gemini图像生成对话，下次需要时请使用命令重新开始")
                e_context["channel"].send(reply, e_context["context"])
                e_context.action = EventAction.BREAK_PASS
            else:
                # 没有活跃会话
                reply = Reply(ReplyType.TEXT, "您当前没有活跃的Gemini图像生成对话")
                e_context["channel"].send(reply, e_context["context"])
                e_context.action = EventAction.BREAK_PASS
            return

        # 检查是否是生成图片命令
        for cmd in self.commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供描述内容，格式：{cmd} [描述]")
                    e_context["channel"].send(reply, e_context["context"])
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["channel"].send(reply, e_context["context"])
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 尝试生成图片
                try:
                    # 发送处理中消息
                    processing_reply = Reply(ReplyType.TEXT, "正在调用gemini生成图片，请稍候...")
                    e_context["channel"].send(processing_reply, e_context["context"])
                    
                    # 初始化会话状态
                    if conversation_key not in self.conversations:
                        self.conversations[conversation_key] = []
                        self.conversation_session_types[conversation_key] = self.SESSION_TYPE_GENERATE
                        self.last_conversation_time[conversation_key] = time.time()
                    
                    # 获取上下文历史
                    conversation_history = self.conversations[conversation_key]
                    
                    # 翻译提示词
                    translated_prompt = self._translate_prompt(prompt, user_id)
                    
                    # 生成图片
                    image_datas, text_responses = self._generate_image(prompt, conversation_history)

                    
                    if image_datas:
                        # 在生成图片之前确保clean_texts有效
                        if text_responses and any(text is not None for text in text_responses):
                            # 过滤掉None值
                            valid_responses = [text for text in text_responses if text]
                            if valid_responses:
                                clean_texts = [text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_") for text in valid_responses]
                                clean_texts = [text[:30] + "..." if len(text) > 30 else text for text in clean_texts]
                            else:
                                clean_texts = ["generated_image"]  # 默认名称
                        else:
                            clean_texts = ["generated_image"]  # 默认名称
                        
                        # 保存图片到本地
                        image_paths = []
                        for i, image_data in enumerate(image_datas):
                            if image_data is not None:  # 确保图片数据不为None
                                # 确保有足够的clean_text
                                clean_text = clean_texts[i] if i < len(clean_texts) else f"image_{i}"
                                image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                                image_paths.append(image_path)
                        
                        # 只有在成功保存了图片时才更新和处理会话
                        if image_paths:
                            # 保存最后生成的图片路径
                            self.last_images[conversation_key] = image_paths
                            
                            # 添加用户提示到会话
                            user_messages = [{"role": "user", "parts": [{"text": prompt}]} for prompt in prompt.split()]
                            conversation_history.extend(user_messages)
                            
                            # 添加助手回复到会话
                            assistant_messages = [
                                {
                                    "role": "model", 
                                    "parts": [
                                        {"text": text_response if text_response else "图片生成成功！"},
                                        {"image_url": image_path}
                                    ]
                                }
                                for text_response, image_path in zip(text_responses, image_paths)
                            ]
                            conversation_history.extend(assistant_messages)
                            
                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]
                            
                            # 更新会话时间戳
                            self.last_conversation_time[conversation_key] = time.time()
                            
                            # 先发送文本消息
                            has_sent_text = False
                            for i, (text_response, image_data) in enumerate(zip(text_responses, image_datas)):
                                if text_response:  # 如果有文本，先发送文本
                                    e_context["channel"].send(Reply(ReplyType.TEXT, text_response), e_context["context"])
                                    has_sent_text = True  # 标记已发送文本
                                
                                if image_data:  # 如果有图片，再发送图片
                                    # 创建临时文件保存图片，每个图片都需要单独发送
                                    temp_image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                    with open(temp_image_path, "wb") as f:
                                        f.write(image_data)
                                    
                                    # 单独发送每张图片
                                    image_file = open(temp_image_path, "rb")
                                    e_context["channel"].send(Reply(ReplyType.IMAGE, image_file), e_context["context"])
                            
                            # 如果已经发送了文本，则不再重复发送
                            if not has_sent_text:
                                # 只有在没有发送过文本的情况下，才发送汇总文本
                                if any(text is not None for text in text_responses):
                                    valid_responses = [text for text in text_responses if text]
                                    if valid_responses:
                                        translated_responses = [self._translate_gemini_message(text) for text in valid_responses]
                                        reply_text = "\n".join([resp for resp in translated_responses if resp])
                                        e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                                else:
                                    # 检查是否有文本响应，可能是内容被拒绝
                                    if text_responses and any(text is not None for text in text_responses):
                                        # 过滤掉None值
                                        valid_responses = [text for text in text_responses if text]
                                        if valid_responses:
                                            # 内容审核拒绝的情况，翻译并发送拒绝消息
                                            translated_responses = [self._translate_gemini_message(text) for text in valid_responses]
                                            reply_text = "\n".join([resp for resp in translated_responses if resp])
                                            e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                                        else:
                                            e_context["channel"].send(Reply(ReplyType.TEXT, "图片生成失败，请稍后再试或修改提示词"), e_context["context"])
                            # 确保只设置一次action
                            e_context.action = EventAction.BREAK_PASS
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_responses and any(text is not None for text in text_responses):
                            # 过滤掉None值
                            valid_responses = [text for text in text_responses if text]
                            if valid_responses:
                                # 内容审核拒绝的情况，翻译并发送拒绝消息
                                translated_responses = [self._translate_gemini_message(text) for text in valid_responses]
                                reply_text = "\n".join([resp for resp in translated_responses if resp])
                                e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                            else:
                                e_context["channel"].send(Reply(ReplyType.TEXT, "图片生成失败，请稍后再试或修改提示词"), e_context["context"])
                            e_context.action = EventAction.BREAK_PASS
                        else:
                            # 没有有效的文本响应或图片，返回一个通用错误消息并中断处理
                            e_context["channel"].send(Reply(ReplyType.TEXT, "图片生成失败，请稍后再试或修改提示词"), e_context["context"])
                            e_context.action = EventAction.BREAK_PASS
                except Exception as e:
                    logger.error(f"生成图片失败: {str(e)}")
                    logger.exception(e)
                    reply_text = f"生成图片失败: {str(e)}"
                    e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                    # 确保在异常情况下也设置正确的action，防止命令继续传递
                    e_context.action = EventAction.BREAK_PASS
                return

        # 检查是否是编辑图片命令
        for cmd in self.edit_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供编辑描述，格式：{cmd} [描述]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 先尝试从缓存获取最近的图片
                image_data = self._get_recent_image(conversation_key)
                if image_data:
                    # 如果找到缓存的图片，保存到本地再处理
                    image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    self.last_images[conversation_key] = image_path
                    logger.info(f"找到最近缓存的图片，保存到：{image_path}")
                    
                    # 尝试编辑图片
                    try:
                        # 发送处理中消息
                        processing_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                        e_context["reply"] = processing_reply
                        
                        # 获取会话上下文
                        conversation_history = self.conversations[conversation_key]
                        
                        # 翻译提示词
                        translated_prompt = self._translate_prompt(prompt, user_id)
                        
                        # 编辑图片
                        result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
                        
                        if result_image:
                            # 保存编辑后的图片
                            reply_text = text_response if text_response else "图片编辑成功！"
                            if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                            
                            # 将回复文本添加到文件名中
                            clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                            clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                            
                            image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                            with open(image_path, "wb") as f:
                                f.write(result_image)
                            
                            # 保存最后生成的图片路径
                            self.last_images[conversation_key] = image_path
                            
                            # 添加用户提示到会话
                            user_message = {"role": "user", "parts": [{"text": prompt}]}
                            conversation_history.append(user_message)
                            
                            # 添加助手回复到会话
                            assistant_message = {
                                "role": "model", 
                                "parts": [
                                    {"text": text_response if text_response else "图片编辑成功！"},
                                    {"image_url": image_path}
                                ]
                            }
                            conversation_history.append(assistant_message)
                            
                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]
                            
                            # 更新会话时间戳
                            self.last_conversation_time[conversation_key] = time.time()
                            
                            # 准备回复文本
                            reply_text = text_response if text_response else "图片编辑成功！"
                            if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                            
                            # 先发送文本消息
                            e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                            
                            # 创建文件对象，由框架负责关闭
                            image_file = open(image_path, "rb")
                            e_context["reply"] = Reply(ReplyType.IMAGE, image_file)
                            e_context.action = EventAction.BREAK_PASS
                        else:
                            # 检查是否有文本响应，可能是内容被拒绝
                            if text_response:
                                # 内容审核拒绝的情况，翻译并发送拒绝消息
                                translated_response = self._translate_gemini_message(text_response)
                                reply = Reply(ReplyType.TEXT, translated_response)
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                            else:
                                reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改提示词")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                    except Exception as e:
                        logger.error(f"编辑图片失败: {str(e)}")
                        logger.exception(e)
                        reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                    return
                else:
                    # 没有找到缓存的图片，检查是否有最后生成的图片
                    if conversation_key in self.last_images:
                        last_image_path = self.last_images[conversation_key]
                        # 确保last_image_path是字符串类型
                        if isinstance(last_image_path, list):
                            last_image_path = last_image_path[0] if last_image_path else None
                        if last_image_path and os.path.exists(last_image_path):
                            try:
                                # 发送处理中消息
                                processing_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                                e_context["reply"] = processing_reply
                                
                                # 读取图片数据
                                with open(last_image_path, "rb") as f:
                                    image_data = f.read()
                                
                                # 获取会话上下文
                                conversation_history = self.conversations[conversation_key]
                                
                                # 翻译提示词
                                translated_prompt = self._translate_prompt(prompt, user_id)
                                
                                # 编辑图片
                                result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
                                
                                if result_image:
                                    # 保存编辑后的图片
                                    reply_text = text_response if text_response else "图片编辑成功！"
                                    
                                    # 将回复文本添加到文件名中
                                    clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                                    clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                                    
                                    image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                                    with open(image_path, "wb") as f:
                                        f.write(result_image)
                                    
                                    # 保存最后生成的图片路径
                                    self.last_images[conversation_key] = image_path
                                    
                                    # 添加用户提示到会话
                                    user_message = {"role": "user", "parts": [{"text": prompt}]}
                                    conversation_history.append(user_message)
                                    
                                    # 添加助手回复到会话
                                    assistant_message = {
                                        "role": "model", 
                                        "parts": [
                                            {"text": text_response if text_response else "图片编辑成功！"},
                                            {"image_url": image_path}
                                        ]
                                    }
                                    conversation_history.append(assistant_message)
                                    
                                    # 限制会话历史长度
                                    if len(conversation_history) > 10:  # 保留最近5轮对话
                                        conversation_history = conversation_history[-10:]
                                    
                                    # 更新会话时间戳
                                    self.last_conversation_time[conversation_key] = time.time()
                                    
                                    # 准备回复文本
                                    reply_text = text_response if text_response else "图片编辑成功！"
                                    if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                        reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                                    
                                    # 先发送文本消息
                                    e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                                    
                                    # 创建文件对象，由框架负责关闭
                                    image_file = open(image_path, "rb")
                                    e_context["reply"] = Reply(ReplyType.IMAGE, image_file)
                                    e_context.action = EventAction.BREAK_PASS
                                else:
                                    # 检查是否有文本响应，可能是内容被拒绝
                                    if text_response:
                                        # 内容审核拒绝的情况，翻译并发送拒绝消息
                                        translated_response = self._translate_gemini_message(text_response)
                                        reply = Reply(ReplyType.TEXT, translated_response)
                                        e_context["reply"] = reply
                                        e_context.action = EventAction.BREAK_PASS
                                    else:
                                        reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改提示词")
                                        e_context["reply"] = reply
                                        e_context.action = EventAction.BREAK_PASS
                            except Exception as e:
                                logger.error(f"编辑图片失败: {str(e)}")
                                logger.exception(e)
                                reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                            return
                        else:
                            # 图片文件已丢失
                            reply = Reply(ReplyType.TEXT, "找不到之前生成的图片，请重新生成图片后再编辑")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                    else:
                        # 没有之前生成的图片
                        reply = Reply(ReplyType.TEXT, "请先使用生成图片命令生成一张图片，或者上传一张图片后再编辑")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                        
        # 检查是否是参考图编辑命令
        for cmd in self.reference_edit_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供编辑描述，格式：{cmd} [描述]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查当前会话类型，无论是什么类型都重置会话（参考图编辑总是新的会话）
                self._create_or_reset_conversation(conversation_key, self.SESSION_TYPE_REFERENCE, False)
                
                # 记录用户正在等待上传参考图片
                self.waiting_for_reference_image[user_id] = prompt
                self.waiting_for_reference_image_time[user_id] = time.time()
                
                # 记录日志
                logger.info(f"用户 {user_id} 开始等待上传参考图片，提示词: {prompt}")
                
                # 发送提示消息
                reply = Reply(ReplyType.TEXT, "请发送需要gemini编辑的参考图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return

        # 检查是否是融图命令
        for cmd in self.merge_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供融图描述，格式：{cmd} [描述]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 记录用户正在等待上传融图的第一张图片
                self.waiting_for_merge_image[user_id] = prompt
                self.waiting_for_merge_image_time[user_id] = time.time()
                
                # 记录日志
                logger.info(f"用户 {user_id} 开始等待上传融图的第一张图片，提示词: {prompt}")
                
                # 发送提示消息
                reply = Reply(ReplyType.TEXT, "请发送需要gemini融图的第一张图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return

    def _handle_image_message(self, e_context: EventContext):
        """处理图片消息，缓存图片数据以备后续编辑使用"""
        context = e_context['context']
        session_id = context.get("session_id")
        is_group = context.get("isgroup", False)
        
        # 获取图片内容路径
        image_path = context.content
        logger.info(f"收到图片消息，路径: {image_path}")
        
        # 获取发送者ID，确保群聊和单聊场景都能正确缓存
        sender_id = context.get("from_user_id")  # 默认使用from_user_id
        
        if 'msg' in context.kwargs:
            msg = context.kwargs['msg']
            
            # 在群聊中，优先使用actual_user_id作为用户标识
            if is_group and hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                sender_id = msg.actual_user_id
                logger.info(f"群聊中使用actual_user_id作为发送者ID: {sender_id}")
            elif not is_group:
                # 私聊中使用from_user_id或session_id
                if hasattr(msg, 'from_user_id') and msg.from_user_id:
                    sender_id = msg.from_user_id
                    logger.info(f"私聊中使用from_user_id作为发送者ID: {sender_id}")
                else:
                    sender_id = session_id
                    logger.info(f"私聊中使用session_id作为发送者ID: {sender_id}")
            
            # 使用统一的图片获取方法获取图片数据
            logger.info(f"开始获取图片数据，图片路径: {image_path}, 发送者ID: {sender_id}")
            image_data = self._get_image_data(msg, image_path)
            
            # 如果获取到图片数据，进行处理
            if image_data and len(image_data) > 1000:  # 确保数据大小合理
                try:
                    # 验证是否为有效的图片数据
                    Image.open(BytesIO(image_data))
                    
                    # 保存图片到缓存 - 使用多个键增加找到图片的机会
                    self.image_cache[session_id] = {
                        "content": image_data,
                        "timestamp": time.time()
                    }
                    
                    # 如果sender_id存在且与session_id不同，也用sender_id缓存
                    if sender_id and sender_id != session_id:
                        self.image_cache[sender_id] = {
                            "content": image_data,
                            "timestamp": time.time()
                        }
                    
                    # 修复日志记录格式    
                    log_message = f"成功缓存图片数据，大小: {len(image_data)} 字节，缓存键: {session_id}"
                    if sender_id and sender_id != session_id:
                        log_message += f", {sender_id}"
                    logger.info(log_message)
                    
                    # 检查是否有用户在等待上传参考图片
                    if sender_id and sender_id in self.waiting_for_reference_image:
                        prompt = self.waiting_for_reference_image[sender_id]
                        logger.info(f"检测到用户 {sender_id} 正在等待上传参考图片，提示词: {prompt}")
                        
                        # 将图片转换为base64
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # 清除等待状态
                        del self.waiting_for_reference_image[sender_id]
                        if sender_id in self.waiting_for_reference_image_time:
                            del self.waiting_for_reference_image_time[sender_id]
                        
                        # 直接发送成功获取图片的提示
                        processing_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                        e_context["reply"] = processing_reply
                        e_context.action = EventAction.BREAK_PASS
                        e_context["channel"].send(processing_reply, e_context["context"])
                        
                        # 处理参考图片编辑
                        self._handle_reference_image_edit(e_context, sender_id, prompt, image_base64)
                        return
                    # 检查是否有用户在等待反推提示词
                    elif sender_id and sender_id in self.waiting_for_reverse_image:
                        # 检查是否超时
                        if time.time() - self.waiting_for_reverse_image_time[sender_id] > self.reverse_image_wait_timeout:
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, "图片上传超时，请重新发送反推提示词命令")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        
                        try:
                            # 调用API分析图片
                            logger.info(f"开始反推提示词，图片大小: {len(image_data)} 字节")
                            reverse_result = self._reverse_image(image_data)
                            if reverse_result:
                                logger.info(f"反推提示词成功，结果长度: {len(reverse_result)}")
                                reply = Reply(ReplyType.TEXT, reverse_result)
                            else:
                                logger.error("反推提示词失败，API返回为空")
                                reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                            
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        except Exception as e:
                            logger.error(f"处理反推请求异常: {str(e)}")
                            logger.exception(e)
                            
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                    # 检查是否有用户在等待识图
                    elif sender_id and sender_id in self.waiting_for_analysis_image:
                        # 检查是否超时
                        if time.time() - self.waiting_for_analysis_image_time[sender_id] > self.analysis_image_wait_timeout:
                            # 清理状态
                            del self.waiting_for_analysis_image[sender_id]
                            del self.waiting_for_analysis_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, "图片上传超时，请重新发送识图命令")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        
                        try:
                            # 获取用户的问题或默认提示词
                            question = self.waiting_for_analysis_image[sender_id]
                            logger.info(f"开始识图，问题: {question}, 图片大小: {len(image_data)} 字节")
                            
                            # 调用API分析图片
                            analysis_result = self._analyze_image(image_data, question)
                            if analysis_result:
                                logger.info(f"识图成功，结果长度: {len(analysis_result)}")
                                # 缓存图片数据和时间戳，用于后续追问
                                self.last_analysis_image[sender_id] = image_data
                                self.last_analysis_time[sender_id] = time.time()
                                
                                # 添加追问提示
                                analysis_result += "\n💬3min内输入g追问+问题，可继续追问"
                                reply = Reply(ReplyType.TEXT, analysis_result)
                            else:
                                logger.error("识图失败，API返回为空")
                                reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                            
                            # 清理状态
                            del self.waiting_for_analysis_image[sender_id]
                            del self.waiting_for_analysis_image_time[sender_id]
                            
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        except Exception as e:
                            logger.error(f"处理识图请求异常: {str(e)}")
                            logger.exception(e)
                            
                            # 清理状态
                            del self.waiting_for_analysis_image[sender_id]
                            del self.waiting_for_analysis_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                    # 检查是否有用户在等待上传融图图片
                    elif sender_id and sender_id in self.waiting_for_merge_image:
                        # 检查是否超时
                        if time.time() - self.waiting_for_merge_image_time[sender_id] > self.merge_image_wait_timeout:
                            # 清理状态
                            del self.waiting_for_merge_image[sender_id]
                            del self.waiting_for_merge_image_time[sender_id]
                            if sender_id in self.merge_first_image:
                                del self.merge_first_image[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, "图片上传超时，请重新发送融图命令")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        
                        # 将图片转换为base64
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # 检查是否是第一张图片
                        if sender_id not in self.merge_first_image:
                            # 保存第一张图片
                            self.merge_first_image[sender_id] = image_base64
                            logger.info(f"接收到融图第一张图片，用户ID: {sender_id}, 图片大小: {len(image_data)} 字节")
                            
                            # 发送成功获取第一张图片的提示
                            success_reply = Reply(ReplyType.TEXT, "成功获取第一张图片，请发送第二张图片")
                            e_context["reply"] = success_reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        else:
                            # 已有第一张图片，这是第二张图片
                            first_image_base64 = self.merge_first_image[sender_id]
                            prompt = self.waiting_for_merge_image[sender_id]
                            logger.info(f"接收到融图第二张图片，用户ID: {sender_id}, 图片大小: {len(image_data)} 字节，提示词: {prompt}")
                            
                            # 清除等待状态
                            del self.waiting_for_merge_image[sender_id]
                            del self.waiting_for_merge_image_time[sender_id]
                            del self.merge_first_image[sender_id]
                            
                            # 删除成功获取图片的提示消息，直接进行处理
                            # 设置事件状态，但不发送消息
                            e_context.action = EventAction.BREAK_PASS
                            
                            # 处理融图
                            self._handle_merge_images(e_context, sender_id, prompt, first_image_base64, image_base64)
                            return
                    else:
                        logger.info(f"已缓存图片，但用户 {sender_id} 没有等待中的图片操作")
                except Exception as img_err:
                    logger.error(f"图片验证失败: {str(img_err)}")
                    logger.exception(img_err)
                    reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            else:
                logger.error(f"无法获取有效的图片数据，图片路径: {image_path}")
                reply = Reply(ReplyType.TEXT, "无法获取图片数据，请重新上传图片或尝试其他格式。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
    def _get_recent_image(self, conversation_key: str) -> Optional[bytes]:
        """获取最近的图片数据，支持群聊和单聊场景
        
        Args:
            conversation_key: 会话标识，可能是session_id或用户ID
            
        Returns:
            Optional[bytes]: 图片数据或None
        """
        logger.info(f"尝试获取会话 {conversation_key} 的最近图片")
        
        # 尝试直接从缓存获取
        if conversation_key in self.image_cache:
            cache_data = self.image_cache[conversation_key]
            if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                logger.info(f"成功从缓存直接获取图片数据，大小: {len(cache_data['content'])} 字节")
                return cache_data["content"]
        
        # 记录image_cache中的所有键以便于调试
        if self.image_cache:
            cache_keys = list(self.image_cache.keys())
            logger.info(f"当前缓存中的所有键: {cache_keys}")
        else:
            logger.info("当前缓存为空")
        
        # 记录last_images中的所有键和路径以便于调试
        if self.last_images:
            last_image_keys = list(self.last_images.keys())
            logger.info(f"last_images中的所有键: {last_image_keys}")
            
            # 记录last_images中与当前会话键相关的图片路径
            if conversation_key in self.last_images:
                last_image_path = self.last_images[conversation_key]
                # 确保last_image_path是字符串类型
                if isinstance(last_image_path, list):
                    last_image_path = last_image_path[0] if last_image_path else None
                
                if last_image_path:
                    logger.info(f"会话 {conversation_key} 的最后一张图片路径: {last_image_path}, 文件存在: {os.path.exists(last_image_path)}")
                    
                    # 如果last_images中有图片但image_cache中没有，尝试从文件读取并加入缓存
                    if os.path.exists(last_image_path):
                        try:
                            with open(last_image_path, "rb") as f:
                                image_data = f.read()
                                # 加入缓存
                                self.image_cache[conversation_key] = {
                                    "content": image_data,
                                    "timestamp": time.time()
                                }
                                logger.info(f"从最后图片路径读取并加入缓存: {last_image_path}")
                                return image_data
                        except Exception as e:
                            logger.error(f"从文件读取图片失败: {e}")
        else:
            logger.info("last_images为空")
            
        # 尝试从conversation_key直接获取缓存
        cache_data = self.image_cache.get(conversation_key)
        if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
            logger.info(f"从缓存获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {conversation_key}")
            return cache_data["content"]
        
        # 群聊场景：尝试使用当前消息上下文中的发送者ID
        context = e_context['context'] if 'e_context' in locals() else None
        if not context and hasattr(self, 'current_context'):
            context = self.current_context
            
        if context and context.get("isgroup", False):
            sender_id = None
            if 'msg' in context.kwargs:
                msg = context.kwargs['msg']
                # 优先使用actual_user_id或from_user_id
                if hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                    sender_id = msg.actual_user_id
                elif hasattr(msg, 'from_user_id') and msg.from_user_id:
                    sender_id = msg.from_user_id
                # 如果sender_id与session_id相同，尝试其他属性
                if sender_id == context.get("session_id"):
                    if hasattr(msg, 'sender_id') and msg.sender_id:
                        sender_id = msg.sender_id
                    elif hasattr(msg, 'sender_wxid') and msg.sender_wxid:
                        sender_id = msg.sender_wxid
                    elif hasattr(msg, 'self_display_name') and msg.self_display_name:
                        sender_id = msg.self_display_name
                
                if sender_id:
                    # 使用群ID_用户ID格式查找
                    group_key = f"{context.get('session_id')}_{sender_id}"
                    cache_data = self.image_cache.get(group_key)
                    if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                        logger.info(f"从群聊缓存键获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {group_key}")
                        return cache_data["content"]
        
        # 遍历所有缓存键，查找匹配的键
        for cache_key in self.image_cache:
            if cache_key.startswith(f"{conversation_key}_") or cache_key.endswith(f"_{conversation_key}"):
                cache_data = self.image_cache.get(cache_key)
                if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"从组合缓存键获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {cache_key}")
                    return cache_data["content"]
                
        # 如果没有找到，尝试其他方法
        if '_' in conversation_key:
            # 拆分组合键，可能是群ID_用户ID格式
            parts = conversation_key.split('_')
            for part in parts:
                cache_data = self.image_cache.get(part)
                if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"从拆分键部分获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {part}")
                    return cache_data["content"]
                    
        return None
    
    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self.image_cache.items():
            if current_time - cache_data["timestamp"] > self.image_cache_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.image_cache[key]
            logger.debug(f"清理过期图片缓存: {key}")
    
    def _cleanup_expired_conversations(self):
        """清理过期会话"""
        current_time = time.time()
        expired_keys = []
        
        for key, last_time in list(self.last_conversation_time.items()):
            if current_time - last_time > self.conversation_expire_seconds:
                expired_keys.append(key)
                
        for key in expired_keys:
            if key in self.conversations:
                del self.conversations[key]
            if key in self.last_conversation_time:
                del self.last_conversation_time[key]
        
        # 检查并清理过长的会话，防止请求体过大
        for key in list(self.conversations.keys()):
            if isinstance(self.conversations[key], dict) and "messages" in self.conversations[key]:
                messages = self.conversations[key]["messages"]
                if len(messages) > self.MAX_CONVERSATION_MESSAGES:
                    # 保留最近的消息
                    excess = len(messages) - self.MAX_CONVERSATION_MESSAGES
                    self.conversations[key]["messages"] = messages[excess:]
                    logger.info(f"会话 {key} 长度超过限制，已裁剪为最新的 {self.MAX_CONVERSATION_MESSAGES} 条消息")
                
        logger.info(f"已清理 {len(expired_keys)} 个过期会话")
    
    def _safe_api_response_for_logging(self, response_json):
        """
        创建API响应的安全版本，用于日志记录
        将base64数据替换为长度指示器，避免在日志中记录大量数据
        
        Args:
            response_json: 原始API响应JSON
            
        Returns:
            安全版本的API响应，适合记录到日志
        """
        if response_json is None:
            return None
            
        if isinstance(response_json, dict):
            safe_response = {}
            for key, value in response_json.items():
                # 特殊处理可能包含base64数据的字段
                if key == "data" and isinstance(value, str) and len(value) > 100 and self._is_likely_base64(value):
                    safe_response[key] = f"{value[:20]}... [长度: {len(value)}字符]"
                else:
                    safe_response[key] = self._safe_api_response_for_logging(value)
            return safe_response
        elif isinstance(response_json, list):
            return [self._safe_api_response_for_logging(item) for item in response_json]
        elif isinstance(response_json, str) and len(response_json) > 100 and self._is_likely_base64(response_json):
            # 可能是base64编码的数据，只保留前20个字符
            return f"{response_json[:20]}... [长度: {len(response_json)}字符]"
        else:
            return response_json
    
    def _is_likely_base64(self, s):
        """
        判断字符串是否可能是base64编码
        
        Args:
            s: 要检查的字符串
            
        Returns:
            bool: 是否可能是base64编码
        """
        # base64编码通常只包含A-Z, a-z, 0-9, +, /, =
        if not s or len(s) < 50:  # 太短的字符串不太可能是需要截断的base64
            return False
            
        # 检查字符是否符合base64编码
        base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        # 允许少量非base64字符(如换行符)
        non_base64_count = sum(1 for c in s if c not in base64_chars)
        
        # 如果非base64字符比例很低，且字符串很长，则可能是base64编码
        return non_base64_count < len(s) * 0.05 and len(s) > 100  # 允许最多5%的非base64字符
    
    def _chat_with_gemini(self, prompt: str, conversation_history: List[Dict] = None) -> Optional[str]:
        """调用Gemini API进行纯文本对话，返回文本响应"""
        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.chat_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.chat_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            params = {
                "key": self.api_key
            }
        
        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            data = {
                "contents": conversation_history + [{"role": "user", "parts": [{"text": prompt}]}]
            }
        else:
            # 无会话历史，直接发送提示词
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}]
            }
        
        try:
            # 发送请求
            proxies = None
            # 只有在直接调用Google API且启用了代理时才使用代理
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
                response = requests.post(url, headers=headers, params=params, json=data, proxies=proxies)
            else:
                response = requests.post(url, headers=headers, params=params, json=data)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 解析响应数据
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            return parts[0]["text"]
                return None
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"调用Gemini API异常: {str(e)}")
            logger.exception(e)
            return None

    def _expand_prompt(self, prompt: str) -> Optional[str]:
        """扩写提示词
        
        Args:
            prompt: 原始提示词
            
        Returns:
            扩写后的提示词
        """
        # 如果提示词为空，直接返回
        if not prompt or len(prompt.strip()) == 0:
            return prompt
            
        # 获取系统提示词模板和模型
        expand_model = self.config.get("expand_model", "gemini-2.0-flash-thinking-exp-01-21")
        system_prompt = self.config.get("expand_prompt", "请帮我扩写以下提示词，使其更加详细和具体：{prompt}").format(prompt=prompt)
        
        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{expand_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{expand_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            params = {
                "key": self.api_key
            }
        
        # 构建请求数据
        data = {
            "contents": [
                {                    
                    "role": "model",
                    "parts": [{"text": system_prompt}]
                },
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        try:
            # 发送请求
            proxies = None
            # 只有在直接调用Google API且启用了代理时才使用代理
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
                response = requests.post(url, headers=headers, params=params, json=data, proxies=proxies)
            else:
                response = requests.post(url, headers=headers, params=params, json=data)
            
            # 检查响应状态码
            if response.status_code == 200:
                # 解析响应数据
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            return parts[0]["text"]
                return None
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"调用Gemini API异常: {str(e)}")
            logger.exception(e)
            return None

    def _generate_image(self, prompt: str, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API生成图片，返回图片数据和文本响应"""

        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.image_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            params = {
                "key": self.api_key
            }
        
        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg["role"]
                if role == "assistant":
                    role = "model"
                
                processed_msg = {"role": role, "parts": []}
                for part in msg["parts"]:
                    if "text" in part:
                        processed_msg["parts"].append({"text": part["text"]})
                    elif "image_url" in part:
                        # 需要读取图片并转换为inlineData格式
                        try:
                            with open(part["image_url"], "rb") as f:
                                image_data = f.read()
                                image_base64 = base64.b64encode(image_data).decode("utf-8")
                                processed_msg["parts"].append({
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_base64
                                    }
                                })
                        except Exception as e:
                            logger.error(f"处理历史图片失败: {e}")
                            # 跳过这个图片
                processed_history.append(processed_msg)
            
            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        else:
            # 无会话历史，直接使用提示
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        
        # 创建代理配置
        proxies = None
        if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
            # 只有在直接调用Google API且启用了代理时才使用代理
            proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        
        try:
            # 发送请求
            logger.info(f"开始调用Gemini API生成图片")
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=data,
                proxies=proxies,
                timeout=120  # 增加超时时间到120秒
            )
            
            logger.info(f"Gemini API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                # 先记录响应内容，便于调试
                response_text = response.text
                logger.debug(f"Gemini API原始响应内容长度: {len(response_text)}, 前100个字符: {response_text[:100] if response_text else '空'}")
                
                # 检查响应内容是否为空
                if not response_text.strip():
                    logger.error("Gemini API返回了空响应")
                    return None, "API返回了空响应，请检查网络连接或代理服务配置"
                
                try:
                    result = response.json()
                    # 记录解析后的JSON结构
                    logger.debug(f"Gemini API响应JSON结构: {result}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON解析错误: {str(json_err)}, 响应内容: {response_text[:200]}")
                    # 检查是否是代理服务问题
                    if self.use_proxy_service:
                        logger.error("可能是代理服务配置问题，尝试禁用代理服务或检查代理服务实现")
                        return None, "API响应格式错误，可能是代理服务配置问题。请检查代理服务实现或暂时禁用代理服务。"
                    return None, f"API响应格式错误: {str(json_err)}"
                
                # 提取响应
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 处理文本和图片响应，以列表形式返回所有部分
                    text_responses = []
                    image_datas = []
                    
                    for part in parts:
                        # 处理文本部分
                        if "text" in part and part["text"]:
                            text_responses.append(part["text"])
                            image_datas.append(None)  # 对应位置添加None表示没有图片
                        
                        # 处理图片部分
                        elif "inlineData" in part:
                            inline_data = part.get("inlineData", {})
                            if inline_data and "data" in inline_data:
                                # Base64解码图片数据
                                img_data = base64.b64decode(inline_data["data"])
                                image_datas.append(img_data)
                                text_responses.append(None)  # 对应位置添加None表示没有文本
                    
                    if not image_datas or all(img is None for img in image_datas):
                        logger.error(f"API响应中没有找到图片数据: {result}")
                        # 检查是否有文本响应，仅返回文本数据
                        if text_responses and any(text is not None for text in text_responses):
                            # 仅返回文本响应，不修改e_context
                            return [], text_responses  # 返回空图片列表和文本
                        return [], []
                    
                    return image_datas, text_responses
                
                logger.error(f"未找到生成的内容: {result}")
                return [], None, "未找到生成的内容"
            elif response.status_code == 400:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查请求参数或网络连接"
            elif response.status_code == 401:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 403:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 503:
                # 特殊处理503状态码
                try:
                    error_info = response.json()
                    error_message = error_info.get("error", {}).get("message", "服务暂时不可用，请稍后重试")
                    logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {error_message}")
                    return [], None, error_message
                except:
                    return [], None, "服务暂时不可用，请稍后重试"
            elif response.status_code == 429:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请稍后再试或检查代理服务配置"
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查网络连接或代理服务配置".replace('\n', '').replace('\r', '')
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return [], None, f"API调用异常: {str(e)}"
    
    def _edit_image(self, prompt: str, image_data: bytes, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API编辑图片，返回图片数据和文本响应"""
        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.image_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            params = {
                "key": self.api_key
            }
        
        # 将图片数据转换为Base64编码
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 确保msg是字典类型
                if isinstance(msg, str):
                    # 如果是字符串，创建一个简单的文本消息
                    processed_msg = {"role": "user", "parts": [{"text": msg}]}
                    processed_history.append(processed_msg)
                    continue
                
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                
                processed_msg = {"role": role, "parts": []}
                parts = msg.get("parts", [])
                
                # 确保parts是列表类型
                if isinstance(parts, str):
                    parts = [{"text": parts}]
                elif isinstance(parts, dict):
                    parts = [parts]
                
                for part in parts:
                    if isinstance(part, str):
                        processed_msg["parts"].append({"text": part})
                    elif isinstance(part, dict):
                        if "text" in part:
                            processed_msg["parts"].append({"text": part["text"]})
                        elif "image_url" in part:
                            # 需要读取图片并转换为inlineData格式
                            try:
                                with open(part["image_url"], "rb") as f:
                                    img_data = f.read()
                                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                                    processed_msg["parts"].append({
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": img_base64
                                        }
                                    })
                            except Exception as e:
                                logger.error(f"处理历史图片失败: {e}")
                                # 跳过这个图片
                processed_history.append(processed_msg)

            # 构建多模态请求
            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        else:
            # 无会话历史，直接使用提示和图片
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        
        # 创建代理配置
        proxies = None
        if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
            # 只有在直接调用Google API且启用了代理时才使用代理
            proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        
        try:
            # 发送请求
            logger.info(f"开始调用Gemini API编辑图片")
            
            # 添加重试逻辑
            max_retries = 5  # 最大重试次数，总共最多尝试 max_retries+1 次（初始请求 + 重试）
            retry_count = 0
            retry_delay = 1  # 初始重试延迟（秒）
            response = None
            
            while retry_count <= max_retries:
                try:
                    # 计算请求体大小
                    request_data = json.dumps(data)
                    request_size = len(request_data)
                    logger.info(f"Gemini API请求体大小: {request_size} 字节 ({request_size/1024/1024:.2f} MB)")
                    
                    # 检查请求体大小是否超过限制
                    if request_size > self.MAX_REQUEST_SIZE:
                        logger.warning(f"请求体大小 ({request_size/1024/1024:.2f} MB) 超出限制，尝试清理会话历史")
                        
                        # 获取会话键
                        conversation_key = None
                        
                        if conversation_history and len(conversation_history) > 0:
                            # 提取最后一条用户消息
                            last_user_message = None
                            for msg in reversed(conversation_history):
                                if msg.get("role") == "user":
                                    last_user_message = msg
                                    break
                            
                            # 清理会话历史，只保留最后一条用户消息
                            if conversation_key in self.conversations:
                                # 保存会话ID
                                conversation_id = self.conversations[conversation_key].get("conversation_id", "")
                                # 创建新的会话，只保留当前用户的提示词
                                self.conversations[conversation_key] = {
                                    "messages": [{"role": "user", "parts": [{"text": prompt}]}],
                                    "conversation_id": conversation_id
                                }
                                logger.info(f"已重置会话 {conversation_key} 的历史记录，只保留当前提示词")
                                
                                # 重建请求数据，不包含历史
                                data = {
                                    "contents": [
                                        {
                                            "parts": [
                                                {
                                                    "text": prompt
                                                }
                                            ]
                                        }
                                    ],
                                    "generationConfig": {
                                        "responseModalities": ["Text", "Image"],
                                        "temperature": 0.4,
                                        "topP": 0.8,
                                        "topK": 40
                                    }
                                }
                                
                                # 重新计算请求体大小
                                request_data = json.dumps(data)
                                request_size = len(request_data)
                                logger.info(f"重建后的请求体大小: {request_size} 字节 ({request_size/1024/1024:.2f} MB)")
                    
                    response = requests.post(
                        url, 
                        headers=headers, 
                        params=params, 
                        json=data,
                        proxies=proxies,
                        timeout=60  # 增加超时时间到60秒
                    )
                    
                    logger.info(f"Gemini API响应状态码: {response.status_code}")
                    
                    # 如果成功或不是503错误，跳出循环
                    if response.status_code == 200 or response.status_code != 503:
                        break
                    
                    # 如果是503错误且未达到最大重试次数，继续重试
                    if response.status_code == 503 and retry_count < max_retries:
                        logger.warning(f"Gemini API服务过载 (状态码: 503)，将进行重试 ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 10)  # 增加延迟，但最多10秒
                        continue
                    else:
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"请求异常: {str(e)}")
                    if retry_count < max_retries:
                        logger.warning(f"请求异常，将进行重试 ({retry_count+1}/{max_retries})")
                        retry_count += 1
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 10)
                        continue
                    else:
                        raise
            
            # 如果所有重试都失败
            if response is None:
                return None, "API调用失败，所有重试尝试均失败"
                
            if response.status_code == 200:
                # 先记录响应内容，便于调试
                response_text = response.text
                logger.debug(f"Gemini API原始响应内容长度: {len(response_text)}, 前100个字符: {response_text[:100] if response_text else '空'}")
                
                # 检查响应内容是否为空
                if not response_text.strip():
                    logger.error("Gemini API返回了空响应")
                    return None, "API返回了空响应，请检查网络连接或代理服务配置"
                
                try:
                    result = response.json()
                    # 记录解析后的JSON结构（安全版本）
                    safe_result = self._safe_api_response_for_logging(result)
                    logger.debug(f"Gemini API响应JSON结构: {safe_result}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON解析错误: {str(json_err)}, 响应内容: {response_text[:200]}")
                    # 检查是否是代理服务问题
                    if self.use_proxy_service:
                        logger.error("可能是代理服务配置问题，尝试禁用代理服务或检查代理服务实现")
                        return None, "API响应格式错误，可能是代理服务配置问题。请检查代理服务实现或暂时禁用代理服务。"
                    return None, f"API响应格式错误: {str(json_err)}"
                
                # 检查是否有内容安全问题
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    finish_reason = candidates[0].get("finishReason", "")
                    if finish_reason == "IMAGE_SAFETY":
                        logger.warning("Gemini API返回IMAGE_SAFETY，图片内容可能违反安全政策")
                        return None, json.dumps(result)  # 返回整个响应作为错误信息
                    
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 处理文本和图片响应
                    text_response = None
                    image_data = None
                    
                    for part in parts:
                        # 处理文本部分
                        if "text" in part and part["text"]:
                            text_response = part["text"]
                        
                        # 处理图片部分
                        if "inlineData" in part:
                            inlineData = part.get("inlineData", {})
                            if inlineData and "data" in inlineData:
                                # 返回Base64解码后的图片数据
                                image_data = base64.b64decode(inlineData["data"])
                    
                    if not image_data:
                        logger.error(f"API响应中没有找到图片数据: {result}")
                    
                    return image_data, text_response
                
                logger.error(f"未找到编辑后的图片数据: {result}")
                return None, None
            elif response.status_code == 400:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查请求参数或网络连接"
            elif response.status_code == 401:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 403:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 429:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return [], None, "API调用失败，请稍后再试或检查代理服务配置"
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查网络连接或代理服务配置"
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return [], None, f"API调用异常: {str(e)}"
    
    def _translate_gemini_message(self, text: str) -> str:
        """将Gemini API的英文消息翻译成中文"""
        # 内容安全过滤消息
        if "finishReason" in text and "IMAGE_SAFETY" in text:
            return "抱歉，您的请求可能违反了内容安全政策，无法生成或编辑图片。请尝试修改您的描述，提供更为安全、合规的内容。"
        
        # 处理API响应中的特定错误
        if "finishReason" in text:
            return "抱歉，图片处理失败，请尝试其他描述或稍后再试。"
            
        # 常见的内容审核拒绝消息翻译
        if "I'm unable to create this image" in text:
            if "sexually suggestive" in text:
                return "抱歉，我无法创建这张图片。我不能生成带有性暗示或促进有害刻板印象的内容。请提供其他描述。"
            elif "harmful" in text or "dangerous" in text:
                return "抱歉，我无法创建这张图片。我不能生成可能有害或危险的内容。请提供其他描述。"
            elif "violent" in text:
                return "抱歉，我无法创建这张图片。我不能生成暴力或血腥的内容。请提供其他描述。"
            else:
                return "抱歉，我无法创建这张图片。请尝试修改您的描述，提供其他内容。"
        
        # 其他常见拒绝消息
        if "cannot generate" in text or "can't generate" in text:
            return "抱歉，我无法生成符合您描述的图片。请尝试其他描述。"
        
        if "against our content policy" in text:
            return "抱歉，您的请求违反了内容政策，无法生成相关图片。请提供其他描述。"
        
        # 默认情况，原样返回
        return text
    
    def _translate_prompt(self, prompt: str, user_id: str = None) -> str:
        """
        将中文提示词翻译成英文
        
        Args:
            prompt: 原始提示词
            user_id: 用户ID，用于获取用户的翻译设置
            
        Returns:
            翻译后的提示词，如果翻译失败则返回原始提示词
        """
        # 如果提示词为空，直接返回
        if not prompt or len(prompt.strip()) == 0:
            return prompt
            
        # 检查全局翻译设置
        if not self.enable_translate:
            return prompt
            
        # 检查用户个人翻译设置（如果有）
        if user_id is not None and user_id in self.user_translate_settings:
            if not self.user_translate_settings[user_id]:
                return prompt
        
        # 检查API密钥是否配置
        if not self.translate_api_key:
            logger.warning("翻译API密钥未配置，将使用原始提示词")
            return prompt
            
        try:
            # 构建请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.translate_api_key}"
            }
            
            data = {
                "model": self.translate_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的中英翻译专家。你的任务是将用户输入的中文提示词翻译成英文，用于AI图像生成。请确保翻译准确、自然，并保留原始提示词的意图和风格。不要添加任何解释或额外内容，只需提供翻译结果。"
                    },
                    {
                        "role": "user",
                        "content": f"请将以下中文提示词翻译成英文，用于AI图像生成：\n\n{prompt}"
                    }
                ]
            }
            
            # 发送请求
            url = f"{self.translate_api_base}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            # 解析响应
            if response.status_code == 200:
                result = response.json()
                translated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 清理翻译结果，移除可能的引号和多余空格
                translated_text = translated_text.strip('"\'').strip()
                
                if translated_text:
                    logger.info(f"翻译成功: {prompt} -> {translated_text}")
                    return translated_text
            
            logger.warning(f"翻译失败: {response.status_code} {response.text}")
            return prompt
            
        except Exception as e:
            logger.error(f"翻译出错: {str(e)}")
            return prompt
    
    def _load_config_template(self):
        """加载配置模板"""
        try:
            template_path = os.path.join(os.path.dirname(__file__), "config.json.template")
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    plugin_conf = json.load(f)
                    return plugin_conf
        except Exception as e:
            logger.exception(e)
            return {
                "enable": True,
                "gemini_api_key": "",
                "model": "gemini-2.0-flash-exp-image-generation",
                "commands": ["g生成图片", "g画图", "g画一个"],
                "edit_commands": ["g编辑图片", "g改图"],
                "reference_edit_commands": ["g参考图", "g编辑参考图"],
                "exit_commands": ["g结束对话", "g结束"],
                "enable_points": False,
                "generate_image_cost": 10,
                "edit_image_cost": 15,
                "save_path": "temp",
                "admins": [],
                "enable_proxy": False,
                "proxy_url": "",
                "use_proxy_service": True,
                "proxy_service_url": "",
                "translate_api_base": "https://open.bigmodel.cn/api/paas/v4",
                "translate_api_key": "",
                "translate_model": "glm-4-flash",
                "enable_translate": True,
                "translate_on_commands": ["g开启翻译", "g启用翻译"],
                "translate_off_commands": ["g关闭翻译", "g禁用翻译"]
            }

    def _get_image_data(self, msg, image_path_or_data):
        """
        统一的图片数据获取方法，参考QwenVision插件的实现
        
        Args:
            msg: 消息对象，可能包含图片数据或路径
            image_path_or_data: 可能是图片路径、URL或二进制数据
            
        Returns:
            bytes: 图片二进制数据，获取失败则返回None
        """
        try:
            # 如果已经是二进制数据，直接返回
            if isinstance(image_path_or_data, bytes):
                logger.debug(f"处理二进制数据，大小: {len(image_path_or_data)} 字节")
                return image_path_or_data
            
            logger.debug(f"开始处理图片，类型: {type(image_path_or_data)}")
            
            # 统一的文件读取函数
            def read_file(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        logger.debug(f"成功读取文件: {file_path}, 大小: {len(data)} 字节")
                        return data
                except Exception as e:
                    logger.error(f"读取文件失败 {file_path}: {e}")
                    return None
            
            # 按优先级尝试不同的读取方式
            # 1. 如果是文件路径，直接读取
            if isinstance(image_path_or_data, str):
                if os.path.isfile(image_path_or_data):
                    data = read_file(image_path_or_data)
                    if data:
                        return data
                
                # 2. 处理URL，尝试下载
                if image_path_or_data.startswith(('http://', 'https://')):
                    try:
                        logger.debug(f"尝试从URL下载图片: {image_path_or_data}")
                        response = requests.get(image_path_or_data, timeout=10)
                        if response.status_code == 200:
                            data = response.content
                            if data and len(data) > 1000:
                                logger.debug(f"从URL下载图片成功，大小: {len(data)} 字节")
                                return data
                    except Exception as e:
                        logger.error(f"从URL下载图片失败: {e}")
                
                # 3. 尝试不同的路径组合
                if image_path_or_data.startswith('tmp/') and not os.path.exists(image_path_or_data):
                    # 尝试使用项目目录
                    project_path = os.path.join(os.path.dirname(__file__), image_path_or_data)
                    if os.path.exists(project_path):
                        data = read_file(project_path)
                        if data:
                            return data
                    
                    # 尝试使用临时目录
                    temp_path = os.path.join("temp", os.path.basename(image_path_or_data))
                    if os.path.exists(temp_path):
                        data = read_file(temp_path)
                        if data:
                            return data
            
            # 4. 从msg对象获取图片数据
            if msg:
                # 4.1 检查file_path属性
                if hasattr(msg, 'file_path') and msg.file_path:
                    file_path = msg.file_path
                    logger.debug(f"从msg.file_path获取到文件路径: {file_path}")
                    data = read_file(file_path)
                    if data:
                        return data
                
                # 4.2 检查msg.content
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, bytes):
                        logger.debug(f"使用msg.content中的二进制内容，大小: {len(msg.content)} 字节")
                        return msg.content
                    elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                        data = read_file(msg.content)
                        if data:
                            return data
                
                # 4.3 尝试使用download_image方法
                if hasattr(msg, 'download_image') and callable(getattr(msg, 'download_image')):
                    try:
                        logger.debug("尝试使用msg.download_image()方法获取图片")
                        image_data = msg.download_image()
                        if image_data and len(image_data) > 1000:
                            logger.debug(f"通过download_image方法获取到图片数据，大小: {len(image_data)} 字节")
                            return image_data
                    except Exception as e:
                        logger.error(f"download_image方法调用失败: {e}")
                
                # 4.4 尝试从msg.img获取
                if hasattr(msg, 'img') and msg.img:
                    image_data = msg.img
                    if image_data and len(image_data) > 1000:
                        logger.debug(f"从msg.img获取到图片数据，大小: {len(image_data)} 字节")
                        return image_data
                
                # 4.5 尝试从msg.msg_data获取
                if hasattr(msg, 'msg_data'):
                    try:
                        msg_data = msg.msg_data
                        if isinstance(msg_data, dict) and 'image' in msg_data:
                            image_data = msg_data['image']
                            if image_data and len(image_data) > 1000:
                                logger.debug(f"从msg_data['image']获取到图片数据，大小: {len(image_data)} 字节")
                                return image_data
                        elif isinstance(msg_data, bytes):
                            image_data = msg_data
                            logger.debug(f"从msg_data(bytes)获取到图片数据，大小: {len(image_data)} 字节")
                            return image_data
                    except Exception as e:
                        logger.error(f"获取msg_data失败: {e}")
                
                # 4.6 微信特殊处理：尝试从_rawmsg获取图片路径
                if hasattr(msg, '_rawmsg') and isinstance(msg._rawmsg, dict):
                    try:
                        rawmsg = msg._rawmsg
                        logger.debug(f"获取到_rawmsg: {type(rawmsg)}")
                        
                        # 检查是否有图片文件路径
                        if 'file' in rawmsg and rawmsg['file']:
                            file_path = rawmsg['file']
                            logger.debug(f"从_rawmsg获取到文件路径: {file_path}")
                            data = read_file(file_path)
                            if data:
                                return data
                    except Exception as e:
                        logger.error(f"处理_rawmsg失败: {e}")
                
                # 4.7 尝试从image_url属性获取
                if hasattr(msg, 'image_url') and msg.image_url:
                    try:
                        image_url = msg.image_url
                        logger.debug(f"从msg.image_url获取图片URL: {image_url}")
                        response = requests.get(image_url, timeout=10)
                        if response.status_code == 200:
                            data = response.content
                            if data and len(data) > 1000:
                                logger.debug(f"从image_url下载图片成功，大小: {len(data)} 字节")
                                return data
                    except Exception as e:
                        logger.error(f"从image_url下载图片失败: {e}")
                
                # 4.8 如果文件未下载，尝试下载 (类似QwenVision的_prepare_fn处理)
                if hasattr(msg, '_prepare_fn') and hasattr(msg, '_prepared') and not msg._prepared:
                    logger.debug("尝试调用msg._prepare_fn()下载图片...")
                    try:
                        msg._prepare_fn()
                        msg._prepared = True
                        time.sleep(1)  # 等待文件准备完成
                        
                        # 再次尝试获取内容
                        if hasattr(msg, 'content'):
                            if isinstance(msg.content, bytes):
                                return msg.content
                            elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                                data = read_file(msg.content)
                                if data:
                                    return data
                    except Exception as e:
                        logger.error(f"调用_prepare_fn下载图片失败: {e}")
            
            logger.error(f"无法获取图片数据: {image_path_or_data}")
            return None
            
        except Exception as e:
            logger.error(f"获取图片数据失败: {e}")
            return None

    def _reverse_image(self, image_data: bytes) -> Optional[str]:
        """调用Gemini API分析图片内容"""
        try:
            # 将图片转换为Base64格式
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # 构建请求数据
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "text": self.reverse_prompt
                            }
                        ]
                    }
                ]
            }
            
            # 根据配置决定使用直接调用还是通过代理服务调用
            if self.use_proxy_service and self.proxy_service_url:
                url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.image_model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
                }
                params = {}
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                }
                params = {
                    "key": self.api_key
                }
            
            # 创建代理配置
            proxies = None
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=data,
                proxies=proxies,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 提取文本响应
                    for part in parts:
                        if "text" in part:
                            return part["text"]
                
                return None
            else:
                logger.error(f"图片分析API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"图片分析异常: {str(e)}")
            logger.exception(e)
            return None

    def _analyze_image(self, image_data: bytes, question: Optional[str] = None) -> Optional[str]:
        """分析图片内容或回答关于图片的问题
        
        Args:
            image_data: 图片二进制数据
            question: 可选，用户关于图片的具体问题
            
        Returns:
            str: 分析结果或问题的回答
        """
        try:
            # 将图片数据转换为base64格式
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 构建请求数据
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            # 如果有具体问题，添加到请求中
            if question:
                data["contents"][0]["parts"].append({"text": question})
            else:
                # 使用默认的分析提示词
                default_prompt = "请仔细观察这张图片的内容，然后用简洁清晰的中文回答用户的问题。如用户没有提出额外问题，则简单描述图片中的主体、场景、风格、颜色等关键要素。如果图片包含文字，也请提取出来。"
                data["contents"][0]["parts"].append({"text": default_prompt})
            
            # 根据配置决定使用直接调用还是通过代理服务调用
            if self.use_proxy_service and self.proxy_service_url:
                url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.image_model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
                }
                params = {}
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                }
                params = {
                    "key": self.api_key
                }
            
            # 创建代理配置
            proxies = None
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=data,
                proxies=proxies,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 提取文本响应
                    for part in parts:
                        if "text" in part:
                            return part["text"]
                
                return None
            else:
                logger.error(f"图片分析API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"分析图片失败: {str(e)}")
            logger.exception(e)
            return None

    def _handle_reference_image_edit(self, e_context, user_id, prompt, image_base64):
        """
        处理参考图片编辑
        
        Args:
            e_context: 事件上下文
            user_id: 用户ID
            prompt: 编辑提示词
            image_base64: 图片的base64编码
        """
        try:
            # 获取会话标识
            session_id = e_context["context"].get("session_id")
            conversation_key = session_id or user_id
            
            # 注意：提示消息已在调用此方法前发送，此处不再重复发送
            
            # 检查图片数据是否有效
            if not image_base64 or len(image_base64) < 100:
                logger.error(f"无效的图片数据: {image_base64[:20] if image_base64 else 'None'}")
                reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            logger.info(f"收到有效的图片数据，长度: {len(image_base64)}")
            
            try:
                # 将base64转换为二进制数据
                image_data = base64.b64decode(image_base64)
                logger.info(f"成功解码图片数据，大小: {len(image_data)} 字节")
                
                # 验证图片数据是否有效
                try:
                    Image.open(BytesIO(image_data))
                    logger.info("图片数据验证成功")
                except Exception as img_err:
                    logger.error(f"图片数据无效: {str(img_err)}")
                    reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            except Exception as decode_err:
                logger.error(f"Base64解码失败: {str(decode_err)}")
                reply = Reply(ReplyType.TEXT, "图片数据解码失败，请重新上传图片。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 确保会话已设置为参考图编辑类型
            if self.conversation_session_types.get(conversation_key) != self.SESSION_TYPE_REFERENCE:
                self._create_or_reset_conversation(conversation_key, self.SESSION_TYPE_REFERENCE, False)
            
            # 获取会话历史
            conversation_history = self.conversations.get(conversation_key, {}).get("messages", [])
            
            # 翻译提示词
            translated_prompt = self._translate_prompt(prompt, user_id)
            logger.info(f"翻译后的提示词: {translated_prompt}")
            
            # 编辑图片
            logger.info("开始调用_edit_image方法")
            result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
            
            if result_image:
                logger.info(f"图片编辑成功，结果大小: {len(result_image)} 字节")
                # 保存编辑后的图片
                reply_text = text_response if text_response else "参考图片编辑成功！"
                if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                    reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                
                # 将回复文本添加到文件名中
                clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                
                image_path = os.path.join(self.save_dir, f"gemini_ref_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                with open(image_path, "wb") as f:
                    f.write(result_image)
                
                # 保存最后生成的图片路径
                self.last_images[conversation_key] = image_path
                
                # 添加用户提示和参考图到会话
                self._add_message_to_conversation(conversation_key, "user", [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }}
                ])
                
                # 添加助手回复到会话
                self._add_message_to_conversation(conversation_key, "model", [
                    {"text": text_response if text_response else "参考图片编辑成功！"},
                    {"image_url": image_path}
                ])
                
                # 限制会话历史长度
                if len(conversation_history) > 10:  # 保留最近5轮对话
                    conversation_history = conversation_history[-10:]
                
                # 更新会话时间戳
                # 确保使用正确的变量名，避免引用不存在的last_conversation_time
                try:
                    self.last_conversation_time[conversation_key] = time.time()
                except Exception as e:
                    logger.error(f"更新会话时间戳失败: {str(e)}")
                    # 如果出错，尝试创建变量
                    if not hasattr(self, 'last_conversation_time'):
                        self.last_conversation_time = {}
                    self.last_conversation_time[conversation_key] = time.time()  # 使用last_conversation_time而非last_conversation_time
                
                # 准备回复文本
                reply_text = text_response if text_response else "参考图片编辑成功！"
                if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                    reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                
                # 先发送文本消息
                e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                
                # 创建文件对象，由框架负责关闭
                image_file = open(image_path, "rb")
                e_context["reply"] = Reply(ReplyType.IMAGE, image_file)
                e_context.action = EventAction.BREAK_PASS
            else:
                logger.error(f"图片编辑失败，API响应: {text_response}")
                # 检查是否有文本响应，可能是内容被拒绝
                if text_response:
                    # 内容审核拒绝的情况，翻译并发送拒绝消息
                    translated_response = self._translate_gemini_message(text_response)
                    reply = Reply(ReplyType.TEXT, translated_response)
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                else:
                    reply = Reply(ReplyType.TEXT, "参考图片编辑失败，请稍后再试或修改提示词")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
        except Exception as e:
            logger.error(f"处理参考图片编辑失败: {str(e)}")
            logger.exception(e)
            reply = Reply(ReplyType.TEXT, f"处理参考图片失败: {str(e)}")
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS
    
    def get_help_text(self, verbose=False, **kwargs):
        help_text = "基于Google Gemini的图像生成插件\n"
        help_text += "可以生成和编辑图片，支持连续对话\n\n"
        help_text += "使用方法：\n"
        help_text += f"1. 生成图片：发送 {self.commands[0]} + 描述，例如：{self.commands[0]} 一只可爱的猫咪\n"
        help_text += f"2. 编辑图片：发送 {self.edit_commands[0]} + 描述，例如：{self.edit_commands[0]} 给猫咪戴上帽子\n"
        help_text += f"3. 参考图编辑：发送 {self.reference_edit_commands[0]} + 描述，然后上传图片\n"
        help_text += f"4. 融图：发送 {self.merge_commands[0]} + 描述，然后按顺序上传两张图片\n"
        help_text += f"5. 识图：发送 {self.image_analysis_commands[0]} 然后上传图片，或发送问题后上传图片\n"
        help_text += f"6. 反推提示：发送 {self.image_reverse_commands[0]} 然后上传图片，可分析图片内容并反推提示词\n"
        help_text += f"7. 追问：发送 {self.follow_up_commands[0]} + 问题，对已识别的图片进行追加提问\n"
        help_text += f"8. 结束对话：发送 {self.exit_commands[0]}\n"
        help_text += f"9. 提示增强：发送 {self.expand_commands[0]} + 绘画提示，可对提示词进行智能扩写\n"
        help_text += f"10. 文本对话：发送 {self.chat_commands[0]} + 问题，可直接进行文本对话\n"
        help_text += f"11. 模型管理：发送 {self.print_model_commands[0]} 查看可用对话模型，发送 {self.switch_model_commands[0]} 切换对话模型\n\n"
        
        if self.enable_translate:
            help_text += "特色功能：\n"
            help_text += "* 前置翻译：所有以g开头的指令会自动将中文提示词翻译成英文，然后再调用Gemini API进行图像生成或编辑，提高生成质量\n"
            help_text += f"* 开启翻译：发送 {self.translate_on_commands[0]} 可以开启前置翻译功能\n"
            help_text += f"* 关闭翻译：发送 {self.translate_off_commands[0]} 可以关闭前置翻译功能\n\n"
        
        if verbose:
            help_text += "配置说明：\n"
            help_text += "* 在config.json中可以自定义触发命令和其他设置\n"
            help_text += "* 可以设置代理或代理服务，解决网络访问问题\n"
            
            if self.enable_translate:
                help_text += "* 可以通过enable_translate选项开启或关闭前置翻译功能\n"
                help_text += "* 每个用户可以单独控制是否启用翻译功能\n"
            
            help_text += "\n注意事项：\n"
            help_text += "* 图片生成可能需要一些时间，请耐心等待\n"
            help_text += "* 会话有效期为3分钟，超时后需要重新开始\n"
            help_text += "* 不支持生成违反内容政策的图片\n"
            help_text += "* 识图和追问功能的等待时间为3分钟\n"
            help_text += "* 追问功能仅在最近一次识图后的3分钟内有效\n"
        
        return help_text
    def _handle_merge_images(self, e_context: EventContext, user_id: str, prompt: str, first_image_base64: str, second_image_base64: str) -> None:
        """
        处理融图请求
        
        Args:
            e_context: 事件上下文
            user_id: 用户ID
            prompt: 提示词
            first_image_base64: 第一张图片的base64数据
            second_image_base64: 第二张图片的base64数据
        """
        channel = e_context["channel"]
        context = e_context["context"]
        
        try:
            # 发送唯一的处理中消息
            processing_reply = Reply(ReplyType.TEXT, "成功获取第二张图片，正在融合中...")
            channel.send(processing_reply, context)
            
            # 确保会话存在并设置为融图模式
            conversation_key = user_id
            self._create_or_reset_conversation(conversation_key, self.SESSION_TYPE_MERGE, False)
            
            # 增强提示词，明确要求生成图片
            enhanced_prompt = f"{prompt}。请生成一张融合两张输入图片的新图片，确保在回复中包含图片。"
            
            # 压缩图片以减小请求体大小
            try:
                # 将base64字符串转换回图像数据
                first_image_data = base64.b64decode(first_image_base64)
                second_image_data = base64.b64decode(second_image_base64)
                
                # 获取原始大小用于日志
                first_size = len(first_image_data)
                second_size = len(second_image_data)
                total_size = first_size + second_size
                
                # 只在图片太大时进行轻度压缩，保留高质量
                max_single_image = 2 * 1024 * 1024  # 单图最大2MB
                max_total_size = 3.5 * 1024 * 1024  # 总大小最大3.5MB (留出空间给其他请求数据)
                
                need_compression = False
                if first_size > max_single_image or second_size > max_single_image or total_size > max_total_size:
                    need_compression = True
                    logger.info(f"图片需要压缩: 第一张{first_size/1024:.1f}KB, 第二张{second_size/1024:.1f}KB, 总计{total_size/1024:.1f}KB")
                
                if need_compression:
                    # 使用高质量设置压缩
                    first_image_data = self._compress_image(first_image_data, max_size=1200, quality=95, conversation_key=conversation_key)
                    second_image_data = self._compress_image(second_image_data, max_size=1200, quality=95, conversation_key=conversation_key)
                
                # 重新转换为base64
                first_image_base64_compressed = base64.b64encode(first_image_data).decode("utf-8")
                second_image_base64_compressed = base64.b64encode(second_image_data).decode("utf-8")
                
                if need_compression:
                    logger.info(f"图片压缩：第一张 {len(first_image_base64)} -> {len(first_image_base64_compressed)}，第二张 {len(second_image_base64)} -> {len(second_image_base64_compressed)}")
                else:
                    logger.info(f"使用原始图片质量，无需压缩: 第一张 {len(first_image_base64_compressed)} 字节, 第二张 {len(second_image_base64_compressed)} 字节")
            except Exception as e:
                logger.error(f"处理图片失败: {str(e)}")
                # 如果处理失败，使用原始图片数据
                first_image_base64_compressed = first_image_base64
                second_image_base64_compressed = second_image_base64
            
            # 创建新的零历史请求，而不使用现有会话历史
            zero_history = [
                {
                    "role": "user",
                    "parts": [
                        {"text": enhanced_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": first_image_base64_compressed
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": second_image_base64_compressed
                            }
                        }
                    ]
                }
            ]
            
            # 根据配置决定使用直接调用还是通过代理服务调用
            if self.use_proxy_service and self.proxy_service_url:
                # 使用代理服务调用API
                api_url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.image_model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
                }
                params = {}  # 不需要在URL参数中传递API密钥
                logger.info(f"使用代理服务进行融图请求")
            else:
                # 直接调用Google API
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.api_key}
                logger.info("使用直接API调用进行融图")
            
            # 处理代理
            proxies = None
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            # 使用官方格式构建请求
            request_data = {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": enhanced_prompt},
                        {"inline_data": {
                            "mime_type": "image/png",
                            "data": first_image_base64_compressed
                        }},
                        {"inline_data": {
                            "mime_type": "image/png",
                            "data": second_image_base64_compressed
                        }}
                    ]
                }],
                "generationConfig": {"responseModalities": ["Text", "Image"]}
            }
            
            # 记录安全版本的请求数据（不包含完整base64数据）
            safe_request = copy.deepcopy(request_data)
            for content in safe_request["contents"]:
                for part in content["parts"]:
                    if "inline_data" in part and "data" in part["inline_data"]:
                        part["inline_data"]["data"] = f"[BASE64_DATA_LENGTH: {len(part['inline_data']['data'])}]"
            logger.debug(f"融图API请求数据: {safe_request}")
            logger.info(f"融图请求结构: 1个用户角色对象，包含1个文本部分和{len(request_data['contents'][0]['parts'])-1}个图片部分")
            
            # 发送请求并处理响应
            try:
                max_retries = 10
                retry_count = 0
                retry_delay = 1
                response = None
                
                while retry_count <= max_retries:
                    try:
                        if retry_count > 0:
                            logger.info(f"第{retry_count}次重试融图API请求...")
                        else:
                            if self.use_proxy_service:
                                logger.info(f"通过代理服务进行融图请求: {enhanced_prompt[:100]}...")
                            else:
                                logger.info(f"直接调用Gemini API进行融图: {enhanced_prompt[:100]}...")
                        
                        response = requests.post(
                            api_url, 
                            headers=headers, 
                            params=params, 
                            json=request_data,
                            proxies=proxies,
                            timeout=60
                        )
                        
                        logger.info(f"融图API响应状态码: {response.status_code}")
                        
                        if response.status_code == 200:
                            break
                            
                        # 对特定错误进行重试
                        if response.status_code in [403, 429, 500, 502, 503, 504] and retry_count < max_retries:
                            logger.warning(f"融图API返回状态码 {response.status_code}，将进行重试 ({retry_count+1}/{max_retries})")
                            retry_count += 1
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)  # 指数退避策略
                            continue
                        elif response.status_code == 400:
                            # 对400错误进行详细记录，这通常表示请求格式有问题
                            try:
                                error_detail = response.json()
                                logger.error(f"融图API返回400错误，详细信息: {error_detail}")
                            except Exception as json_err:
                                logger.error(f"融图API返回400错误，但无法解析响应体: {response.text[:500]}")
                            break
                        else:
                            break
                            
                    except requests.exceptions.RequestException as e:
                        error_msg = str(e)
                        # 去除可能包含API密钥的部分
                        if self.api_key and self.api_key in error_msg:
                            error_msg = error_msg.replace(self.api_key, "[API_KEY]")
                        logger.error(f"融图API请求异常: {error_msg}")
                        
                        if retry_count < max_retries:
                            logger.warning(f"请求异常，将进行重试 ({retry_count+1}/{max_retries})")
                            retry_count += 1
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)
                            continue
                        else:
                            logger.error("已达到最大重试次数，放弃请求")
                            error = "融图请求失败，请稍后再试"
                            image_text_pairs, final_text = [], None
                            break
                
                # 处理最终结果
                if response and response.status_code == 200:
                    result = response.json()
                    # 处理响应结果
                    image_text_pairs, final_text, error = self._process_multi_image_response(result)
                    
                    # 如果没有生成图像，尝试使用英文提示词重试
                    if not image_text_pairs:
                        logger.info("未获取到图像，尝试使用英文提示词重试...")
                        english_prompt = f"Please merge these two images. {prompt}. Make sure to include the generated image in your response."
                        request_data["contents"][0]["parts"][0]["text"] = english_prompt
                        
                        # 记录更新后的请求结构
                        logger.info(f"使用英文提示词重试: '{english_prompt[:100]}...'")
                        safe_request = copy.deepcopy(request_data)
                        for content in safe_request["contents"]:
                            for part in content["parts"]:
                                if "inline_data" in part and "data" in part["inline_data"]:
                                    part["inline_data"]["data"] = f"[BASE64_DATA_LENGTH: {len(part['inline_data']['data'])}]"
                        logger.debug(f"英文提示词融图API请求数据: {safe_request}")
                        
                        # 重新进行请求，使用同样的重试机制
                        retry_count = 0
                        retry_delay = 1
                        
                        while retry_count <= max_retries:
                            try:
                                if retry_count > 0:
                                    logger.info(f"英文提示词第{retry_count}次重试融图API请求...")
                                else:
                                    logger.info("使用英文提示词重试融图请求...")
                                
                                response = requests.post(
                                    api_url, 
                                    headers=headers, 
                                    params=params, 
                                    json=request_data,
                                    proxies=proxies,
                                    timeout=60
                                )
                                
                                logger.info(f"英文提示词融图API响应状态码: {response.status_code}")
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    image_text_pairs, final_text, error = self._process_multi_image_response(result)
                                    break
                                
                                # 对特定错误进行重试
                                if response.status_code in [403, 429, 500, 502, 503, 504] and retry_count < max_retries:
                                    logger.warning(f"英文提示词融图API返回状态码 {response.status_code}，将进行重试 ({retry_count+1}/{max_retries})")
                                    retry_count += 1
                                    time.sleep(retry_delay)
                                    retry_delay = min(retry_delay * 1.5, 10)
                                    continue
                                elif response.status_code == 400:
                                    # 对400错误进行详细记录，这通常表示请求格式有问题
                                    try:
                                        error_detail = response.json()
                                        logger.error(f"英文提示词融图API返回400错误，详细信息: {error_detail}")
                                    except Exception as json_err:
                                        logger.error(f"英文提示词融图API返回400错误，但无法解析响应体: {response.text[:500]}")
                                    break
                                else:
                                    break
                                    
                            except requests.exceptions.RequestException as e:
                                error_msg = str(e)
                                if self.api_key and self.api_key in error_msg:
                                    error_msg = error_msg.replace(self.api_key, "[API_KEY]")
                                logger.error(f"英文提示词融图API请求异常: {error_msg}")
                                
                                if retry_count < max_retries:
                                    logger.warning(f"英文提示词请求异常，将进行重试 ({retry_count+1}/{max_retries})")
                                    retry_count += 1
                                    time.sleep(retry_delay)
                                    retry_delay = min(retry_delay * 1.5, 10)
                                    continue
                                else:
                                    logger.error("英文提示词已达到最大重试次数，放弃请求")
                                    break
                        
                        # 如果英文提示词重试后仍未获取到图像
                        if not image_text_pairs:
                            logger.error("使用英文提示词重试失败，未获取到图像")
                            error = "融图失败，请稍后再试"
                            image_text_pairs, final_text = [], None
                elif response:
                    # 请求失败
                    logger.error(f"融图API请求失败: 状态码 {response.status_code}")
                    
                    # 特殊处理401（未授权）和400（请求格式错误）状态码
                    if response.status_code == 401:
                        error = "融图失败: API密钥无效或未授权，请检查配置"
                        logger.error(f"API密钥验证失败: {response.text[:500]}")
                    elif response.status_code == 400:
                        error = "融图失败: 请求格式错误，请联系开发者"
                        try:
                            error_detail = response.json()
                            # 尝试提取更有用的错误信息
                            if 'error' in error_detail:
                                error_message = error_detail['error'].get('message', '')
                                if error_message:
                                    error = f"融图失败: {error_message}"
                                    logger.error(f"API返回详细错误: {error_message}")
                        except Exception as e:
                            logger.error(f"无法解析错误响应: {response.text[:500]}")
                    else:
                        error = "融图失败，请稍后再试"
                    
                    image_text_pairs, final_text = [], None
                else:
                    # 没有收到响应
                    logger.error("融图API请求未收到响应")
                    error = "融图请求未收到响应，请稍后再试"
                    image_text_pairs, final_text = [], None
                    
            except Exception as e:
                error_msg = str(e)
                # 去除可能包含API密钥的部分
                if self.api_key and self.api_key in error_msg:
                    error_msg = error_msg.replace(self.api_key, "[API_KEY]")
                logger.error(f"融图处理异常: {error_msg}")
                error = "融图失败，请稍后再试或联系管理员"
                image_text_pairs, final_text = [], None
            
            if error:
                logger.error(f"融图失败: {error}")
                error_reply = Reply(ReplyType.TEXT, f"融图失败: {error}")
                channel.send(error_reply, context)
                return
            
            if not image_text_pairs or len(image_text_pairs) == 0:
                logger.warning("API没有返回图片数据，尝试再次调用")
                # 尝试使用英文提示词重试一次
                english_prompt = f"Please merge these two images. {prompt}. Make sure to include the generated image in your response."
                
                # 更新请求数据使用英文提示词
                request_data["contents"][0]["parts"][0]["text"] = english_prompt
                
                # 记录更新后的请求结构
                logger.info(f"第二次尝试使用英文提示词: '{english_prompt[:100]}...'")
                safe_request = copy.deepcopy(request_data)
                for content in safe_request["contents"]:
                    for part in content["parts"]:
                        if "inline_data" in part and "data" in part["inline_data"]:
                            part["inline_data"]["data"] = f"[BASE64_DATA_LENGTH: {len(part['inline_data']['data'])}]"
                logger.debug(f"第二次融图API请求数据: {safe_request}")
                
                try:
                    logger.info(f"使用英文提示词重试融图API调用: {english_prompt[:100]}...")
                    response = requests.post(
                        api_url, 
                        headers=headers, 
                        params=params, 
                        json=request_data,
                        proxies=proxies,
                        timeout=60
                    )
                    
                    logger.info(f"重试融图API响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        # 处理响应结果
                        image_text_pairs, final_text, error = self._process_multi_image_response(result)
                    else:
                        logger.error(f"重试融图API请求失败: {response.status_code}, {response.text}")
                        # 特殊处理400错误码
                        if response.status_code == 400:
                            try:
                                error_detail = response.json()
                                if 'error' in error_detail and 'message' in error_detail['error']:
                                    error = f"融图失败: {error_detail['error']['message']}"
                                    logger.error(f"融图API详细错误: {error_detail['error']['message']}")
                                else:
                                    error = f"融图失败，API返回: {response.status_code}"
                            except Exception as e:
                                error = f"融图失败，API返回: {response.status_code}"
                        else:
                            error = f"融图失败，API返回: {response.status_code}"
                except Exception as e:
                    logger.error(f"重试融图API请求异常: {str(e)}")
                    error = f"融图请求异常: {str(e)}"
                
                # 重试失败后的处理
                if error or not image_text_pairs or len(image_text_pairs) == 0:
                    logger.error("第二次尝试仍未返回图片数据")
                    error_msg = "API未能生成图片，请稍后再试或修改提示词。"
                    if final_text:
                        error_msg += f"\n\nAPI回复: {final_text}"
                    error_reply = Reply(ReplyType.TEXT, error_msg)
                    channel.send(error_reply, context)
                    return
            
            # 发送结果
            logger.info(f"成功获取融图结果，共 {len(image_text_pairs)} 张图片，是否有最终文本: {bool(final_text)}")
            self._send_alternating_content(e_context, image_text_pairs, final_text)
            
            # 将成功的融图操作添加到会话历史中
            if image_text_pairs and len(image_text_pairs) > 0:
                # 添加用户请求到会话历史
                self._add_message_to_conversation(
                    conversation_key,
                    "user",
                    [{"text": enhanced_prompt}]
                )
                
                # 添加模型回复，包含图片和文本
                model_parts = []
                if final_text:
                    model_parts.append({"text": final_text})
                
                # 添加生成的图片到会话历史
                for img_data, img_text in image_text_pairs:
                    if img_text:
                        model_parts.append({"text": img_text})
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    model_parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": img_base64
                        }
                    })
                
                # 添加模型回复到会话历史
                self._add_message_to_conversation(
                    conversation_key,
                    "model",
                    model_parts
                )
            
            # 更新会话时间戳
            self.last_conversation_time[conversation_key] = time.time()
            
        except Exception as e:
            # 安全处理异常信息，避免泄露敏感信息
            error_msg = str(e)
            if self.api_key and self.api_key in error_msg:
                error_msg = error_msg.replace(self.api_key, "[API_KEY]")
            if "generativelanguage.googleapis.com" in error_msg:
                error_msg = error_msg.replace(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.image_model}:generateContent",
                    "API_ENDPOINT"
                )
            if self.proxy_service_url and self.proxy_service_url in error_msg:
                error_msg = error_msg.replace(self.proxy_service_url, "[PROXY_URL]")
            
            logger.error(f"处理融图请求异常: {error_msg}")
            logger.error(traceback.format_exc())
            # 对用户显示友好的错误消息
            error_reply = Reply(ReplyType.TEXT, "融图失败，请稍后再试或联系管理员")
            channel.send(error_reply, context)

    def _process_multi_image_response(self, result: Dict) -> Tuple[List[Tuple[bytes, str]], Optional[str], Optional[str]]:
        """处理多图片响应，返回图片数据、最终文本和错误信息"""
        try:
            candidates = result.get("candidates", [])
            if not candidates or len(candidates) == 0:
                logger.error("未找到生成的图片数据")
                return [], None, "API响应中没有找到有效的数据"
                
            # 检查是否有内容安全问题
            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason == "SAFETY":
                logger.warning("Gemini API因安全原因完成响应")
                safety_message = "内容被安全系统拦截，请修改您的提示词"
                if "text" in candidates[0].get("content", {}).get("parts", [{}])[0]:
                    safety_message += f": {candidates[0]['content']['parts'][0]['text']}"
                return [], None, safety_message
            
            if finish_reason == "RECITATION":
                logger.warning("Gemini API因背诵问题完成响应")
                return [], None, "请更改提示词，避免要求生成复制或违规内容"
                
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                logger.error("API响应中没有parts数据")
                return [], None, "API响应中没有parts数据"
            
            # 收集所有图片和文本对
            image_text_pairs = []
            current_text = ""
            final_text = None
            
            # 调试: 显示所有部分的类型
            part_types = []
            for i, part in enumerate(parts):
                if "text" in part:
                    part_types.append(f"{i+1}:text")
                elif "inlineData" in part:
                    part_types.append(f"{i+1}:image")
                else:
                    part_types.append(f"{i+1}:unknown:{list(part.keys())}")
            logger.debug(f"API响应中的部分类型: {', '.join(part_types)}")
            
            # 处理所有部分
            has_image = False
            for i, part in enumerate(parts):
                logger.debug(f"处理第 {i+1}/{len(parts)} 个part")
                
                # 处理文本部分
                if "text" in part and part["text"]:
                    current_text = part["text"].strip()
                    logger.debug(f"找到文本: {current_text[:50]}...")
                
                # 处理图片部分
                elif "inlineData" in part:
                    inlineData = part.get("inlineData", {})
                    if inlineData and "data" in inlineData:
                        try:
                            # 解码图片数据
                            image_data = base64.b64decode(inlineData["data"])
                            logger.debug(f"成功解码图片数据，大小: {len(image_data)} 字节")
                            
                            # 将当前文本和图片数据配对
                            image_text_pairs.append((image_data, current_text))
                            logger.debug(f"添加图片-文本对 #{len(image_text_pairs)}, 文本长度: {len(current_text)}")
                            
                            # 清空当前文本，准备下一对
                            current_text = ""
                            has_image = True
                        except Exception as e:
                            logger.error(f"解码图片数据失败: {e}")
                            continue
            
            # 检查是否有未处理的最后一段文本
            if current_text:
                logger.debug(f"找到最后一段文本（没有对应图片）: {current_text[:50]}...")
                final_text = current_text
            
            # 如果没有图片但有最终文本，尝试检查文本中是否包含"请稍等"等提示词
            if not has_image and final_text:
                waiting_keywords = ["请稍等", "正在生成", "请等待", "正在处理", "processing", "generating", "please wait", "working on it"]
                if any(keyword in final_text.lower() for keyword in waiting_keywords):
                    logger.warning("API仅返回了等待提示，需要重试")
                    return [], final_text, None
            
            # 记录处理结果
            result_summary = []
            if image_text_pairs:
                result_summary.append(f"{len(image_text_pairs)}张图片")
            if final_text:
                result_summary.append("最后一段文本")
            
            logger.info(f"成功处理: {', '.join(result_summary)}")
            return image_text_pairs, final_text, None
            
        except Exception as e:
            logger.error(f"处理API响应时发生错误: {e}")
            logger.error(traceback.format_exc())
            return [], None, f"处理API响应时发生错误: {e}"

    def _send_alternating_content(self, e_context: EventContext, image_text_pairs: List[Tuple[bytes, str]], final_text: Optional[str]) -> None:
        """
        交替发送文本和图片
        
        Args:
            e_context: 事件上下文
            image_text_pairs: 图片数据和文本对列表 [(image_data, text), ...]
            final_text: 最后的文本内容(可选)
        """
        channel = e_context["channel"]
        context = e_context["context"]
        
        logger.info(f"准备交替发送文本和图片: {len(image_text_pairs)} 个图片, 是否有最终文本: {bool(final_text)}")
        
        # 发送所有图片-文本对
        for i, (image_data, text) in enumerate(image_text_pairs):
            # 发送文本(如果有)
            if text and text.strip():
                logger.info(f"发送第 {i+1}/{len(image_text_pairs)} 对的文本部分，长度: {len(text)}")
                text_reply = Reply(ReplyType.TEXT, text)
                channel.send(text_reply, context)
                time.sleep(0.5)  # 添加小延时确保消息顺序
            
            # 保存并发送图片
            try:
                # 创建临时目录
                temp_dir = TmpDir().path()
                
                # 生成安全的文件名：使用时间戳和随机字符串，避免特殊字符
                timestamp = int(time.time() * 1000)
                random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                # 不使用任何可能包含特殊字符的文本作为文件名
                file_name = f"gemini_image_{timestamp}_{random_str}_{i+1}.png"
                file_path = os.path.join(temp_dir, file_name)
                
                # 保存图片
                with open(file_path, "wb") as f:
                    f.write(image_data)
                
                # 发送图片
                logger.info(f"发送第 {i+1}/{len(image_text_pairs)} 对的图片部分，文件: {file_path}")
                with open(file_path, "rb") as f:
                    img_reply = Reply(ReplyType.IMAGE, f)
                    channel.send(img_reply, context)
                time.sleep(1.0)  # 添加延时确保图片发送完成
            except Exception as e:
                logger.error(f"发送图片失败: {e}")
                error_reply = Reply(ReplyType.TEXT, f"图片发送失败: {str(e)}")
                channel.send(error_reply, context)
        
        # 发送最后的文本(如果有)
        if final_text and final_text.strip():
            logger.info(f"发送最终文本，长度: {len(final_text)}")
            final_reply = Reply(ReplyType.TEXT, final_text)
            channel.send(final_reply, context)
        
        # 设置回复为None，表示已手动处理
        e_context["reply"] = None
        e_context.action = EventAction.BREAK_PASS

    def _compress_image(self, image_data, max_size=800, quality=85, format='JPEG', conversation_key=None):
        """
        压缩图片以减小API请求大小，根据会话长度动态调整压缩参数
        
        Args:
            image_data: 图片数据（字节）
            max_size: 最长边的最大尺寸（像素）
            quality: JPEG压缩质量（1-100）
            format: 输出格式
            conversation_key: 会话键，用于判断会话长度
            
        Returns:
            压缩后的图片数据（字节）
        """
        try:
            import io
            from PIL import Image
            
            # 根据会话长度动态调整压缩参数
            if conversation_key and conversation_key in self.conversations:
                messages_count = len(self.conversations[conversation_key].get("messages", []))
                session_type = self.conversation_session_types.get(conversation_key)
                
                # 融图模式使用更高质量的压缩参数
                if session_type == self.SESSION_TYPE_MERGE:
                    quality = min(quality + 10, 95)  # 融图模式质量提高10%，最高95%
                    max_size = min(max_size + 400, 1200)  # 融图模式最大尺寸增加，最大1200px
                    logger.debug(f"融图模式使用高质量压缩参数: 质量={quality}%, 最大尺寸={max_size}px")
                # 根据会话长度动态降低质量和尺寸
                elif messages_count > 6:
                    quality = max(quality - (messages_count - 6) * 5, 40)  # 每多一轮对话降低5%质量，最低40%
                    max_size = max(max_size - (messages_count - 6) * 50, 500)  # 每多一轮对话降低50像素，最低500
                    logger.info(f"会话长度为{messages_count}轮，自动调整压缩参数：质量={quality}%, 最大尺寸={max_size}px")
                
                # 对参考图模式使用更激进的压缩
                if session_type == self.SESSION_TYPE_REFERENCE:
                    quality = min(quality, 75)  # 参考图编辑模式最高质量75%
                    max_size = min(max_size, 700)  # 参考图编辑模式最大尺寸700px
            
            # 打开图片数据
            img = Image.open(io.BytesIO(image_data))
            original_size = len(image_data)
            original_dimensions = img.size
            
            # 计算新尺寸 - 限制最大尺寸
            width, height = img.size
            new_width, new_height = width, height
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # 调整图片大小
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保存为JPEG格式并压缩
            output = io.BytesIO()
            img.convert('RGB').save(output, format=format, quality=quality, optimize=True)
            compressed_data = output.getvalue()
            compressed_size = len(compressed_data)
            
            # 如果压缩后仍然太大，再次压缩
            if compressed_size > 500 * 1024:  # 如果大于500KB
                # 逐步降低质量直到达到目标大小
                for reduced_quality in [70, 60, 50, 40, 30]:
                    output = io.BytesIO()
                    img.convert('RGB').save(output, format=format, quality=reduced_quality, optimize=True)
                    compressed_data = output.getvalue()
                    compressed_size = len(compressed_data)
                    if compressed_size <= 500 * 1024:
                        break
            
            logger.debug(f"图片压缩: {original_size} 字节 -> {compressed_size} 字节 "
                         f"({compressed_size/original_size:.2%}), "
                         f"尺寸: {original_dimensions[0]}x{original_dimensions[1]} -> {new_width}x{new_height}")
            
            return compressed_data
        except Exception as e:
            logger.error(f"压缩图片时出错: {e}")
            # 如果压缩失败，返回原始图片数据
            return image_data

    def _add_message_to_conversation(self, conversation_key, role, parts):
        """添加消息到会话历史，并进行长度控制
        
        Args:
            conversation_key: 会话ID
            role: 消息的角色 (user/assistant)
            parts: 消息的内容部分
            
        Returns:
            更新后的消息列表
        """
        if conversation_key not in self.conversations:
            self.conversations[conversation_key] = {"messages": [], "conversation_id": ""}
        
        # 添加新消息
        self.conversations[conversation_key]["messages"].append({
            "role": role,
            "parts": parts
        })
        
        # 更新最后交互时间
        self.last_conversation_time[conversation_key] = time.time()
        
        # 控制会话长度，保留最近的消息
        if len(self.conversations[conversation_key]["messages"]) > self.MAX_CONVERSATION_MESSAGES:
            # 移除最旧的消息，保留最新的MAX_CONVERSATION_MESSAGES条
            excess = len(self.conversations[conversation_key]["messages"]) - self.MAX_CONVERSATION_MESSAGES
            self.conversations[conversation_key]["messages"] = self.conversations[conversation_key]["messages"][excess:]
            logger.info(f"会话 {conversation_key} 长度超过限制，已裁剪为最新的 {self.MAX_CONVERSATION_MESSAGES} 条消息")
        
        return self.conversations[conversation_key]["messages"]

    def _create_or_reset_conversation(self, conversation_key: str, session_type: str, preserve_id: bool = False) -> None:
        """创建新会话或重置现有会话
        
        Args:
            conversation_key: 会话标识符
            session_type: 会话类型（使用会话类型常量）
            preserve_id: 是否保留现有会话ID
        """
        # 检查是否需要保留会话ID
        conversation_id = ""
        if preserve_id and conversation_key in self.conversations:
            conversation_id = self.conversations[conversation_key].get("conversation_id", "")
            
        # 创建新的空会话
        self.conversations[conversation_key] = {
            "messages": [],
            "conversation_id": conversation_id
        }
        
        # 更新会话类型和时间戳
        self.conversation_session_types[conversation_key] = session_type
        self.last_conversation_time[conversation_key] = time.time()
        
        logger.info(f"已创建/重置会话 {conversation_key}，类型: {session_type}")
