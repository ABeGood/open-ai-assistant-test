import os
import sys
import json
import time
sys.path.append('..')
sys.path.append(os.path.join(__file__, '..', '..'))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f'BASE DIR: {BASE_DIR}')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
# print(*sys.path, sep='\n')
import config
from AgentToBeNamed.agenttobenamed.data_classes import CodeSnippet as AgentTBNCodeSnippet
import copy
import shutil

import ast
import base64
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET, require_http_methods
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core import serializers as core_serializers
from django.db import models
from rest_framework.serializers import ModelSerializer, SerializerMethodField
from django.utils import timezone
from django.http import JsonResponse
from django.template import RequestContext
from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse
from rest_framework import serializers
from django.contrib.auth import logout

from gui.forms import DataTableForm, AnnotationForm
import pandas as pd
from .models import Conversation, Message, User, CodeSnippet, DataTable
import base64
import imageio.v3 as iio


# ===============================================================
# Agent init; TODO: move somewhere else.
from openai._base_client import SyncHttpxClientWrapper
from langchain_community.chat_models import AzureChatOpenAI
from AgentToBeNamed.agenttobenamed import AgentTBN
from AgentToBeNamed.agenttobenamed.AgentDataFrameManager import AddDataMode


import logging
logger = logging.getLogger(__name__)

try:
    http_proxy = "http://http-proxy2-prg.tech.emea.porsche.biz:3128"
    https_proxy = "http://http-proxy2-prg.tech.emea.porsche.biz:3128"

    proxies = {
        "http://": http_proxy,
        "https://": https_proxy,
    }
    logger.debug(f"Proxy configuration set: {proxies}")
except Exception as e:
    logger.error(f"Failed to configure proxies: {e}", exc_info=True)
    raise

try:
    llm = AzureChatOpenAI(
        azure_endpoint=config.gpt_4o_codriver['OPENAI_API_ENDPOINT'],
        azure_deployment=config.gpt_4o_codriver['DEPLOYMENT_NAME'],
        api_version=config.gpt_4o_codriver['OPENAI_API_VERSION'],
        api_key=config.gpt_4o_codriver['OPENAI_API_KEY'],
        temperature=0
    )
    logger.debug(f"Azure OpenAI client initialized with endpoint: {config.gpt_4o_codriver['OPENAI_API_ENDPOINT']}")
except Exception as e:
    logger.error("Failed to initialize Azure OpenAI client", exc_info=True)
    raise


try:
    from openai._types import Timeout
    from typing import cast

    llm.client._client._client = SyncHttpxClientWrapper(
        base_url=llm.client._client.base_url,
        timeout=cast(Timeout, llm.client._client.timeout),
        proxies=proxies,
        transport=llm.client._client._transport,
        limits=llm.client._client._limits,
        follow_redirects=True,
    )
    logger.debug("Successfully configured SyncHttpxClientWrapper")
except Exception as e:
    logger.error("Failed to configure OpenAI client wrapper", exc_info=True)
    raise

# Agent Initialization
logger.info("AG: Initializing AgentTBN")
try:
    data_specs_path = os.path.join(BASE_DIR, 'data', 'specifications')
    tmp_path = os.path.join(BASE_DIR, 'temp')
    
    # Verify paths exist
    if not os.path.exists(data_specs_path):
        logger.warning(f"Data specifications path does not exist: {data_specs_path}")
        os.makedirs(data_specs_path, exist_ok=True)
        logger.info(f"Created specifications directory: {data_specs_path}")
    if not os.path.exists(tmp_path):
        logger.warning(f"Temp path does not exist: {tmp_path}")
        os.makedirs(tmp_path, exist_ok=True)
        logger.info(f"Created temp directory: {tmp_path}")

    agent_TBN = AgentTBN(
        gpt_model=llm,
        is_query_ok_llm=llm,
        prompt_strategy='hybrid_code_text',
        data_specs_dir_path=data_specs_path,
        tmp_file_path=tmp_path,
        generated_code_exec_timeout=60
    )
    logger.info("Successfully initialized AgentTBN")
except Exception as e:
    logger.error("Failed to initialize AgentTBN", exc_info=True)
    raise

def _get_conversation(request_data) -> Conversation | None:
    """
    Fetches or Creates (If the method is POST) `Conversation` object based on the 
    conversation_id within of the `request`.

    Returns
    -------
    Conversation object or None in case of method not being POST and Conversation
    does not exist.
    """
    logger.debug("Attempting to get conversation from request data")
    
    conversation_id = None
    try:
        conversation_id = int(request_data.POST.get('conversation_id', None))
        logger.debug(f"Retrieved conversation_id from request: {conversation_id}")
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to parse conversation_id: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while fetching conversation_id: {e}", exc_info=True)

    if not isinstance(conversation_id, int):
        logger.debug("No valid conversation_id found")
        conversation_id = None

    try:
        if conversation_id:
            logger.debug(f"Fetching existing conversation with id: {conversation_id}")
            return Conversation.objects.get(id=conversation_id, owner=request_data.user)

        elif request_data.method == 'POST':
            logger.info(f"Creating new conversation for user: {request_data.user.username}")
            return Conversation.objects.create(
                conversation_name='New Conversation',
                owner=request_data.user,
                last_update=timezone.now()
            )
        else:
            logger.debug("No conversation found and method is not POST")
            return None
            
    except Conversation.DoesNotExist:
        logger.warning(f"Conversation not found with id: {conversation_id}")
        return None
    except Exception as e:
        logger.error(f"Error in _get_conversation: {e}", exc_info=True)
        raise

def login_view(request):
    logger.debug("Processing login request")
    logger.info("Processing login request")
    
    csrfContext = RequestContext(request)
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        logger.info(f"Login attempt for user: {username}")

        try:
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                logger.info(f"Successful login for user: {username}")
                messages.success(request, f'Welcome, {username}!')
                return redirect('home')
            else:
                logger.warning(f"Failed login attempt for user: {username}")
                messages.error(request, 'Invalid username or password')
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}", exc_info=True)
            messages.error(request, 'An error occurred during login')
    
    context = {
        'title': 'Login | Assistant Analyst',
    }
    return render(request, 'gui/login.html', context=context)


# def logout_view(request):
#     context = {
#         'title': 'Logout | Assistant Analyst',
#     }
#     return render(request, 'gui/logout.html', context=context)

@csrf_exempt
def logout_view(request):
    logout(request)
    return redirect('login')  # Redirects to the URL with name='login'


def account_request_view(request):
    context = {
        'title': 'New Account Request | Assistant Analyst',
    }
    return render(request, 'gui/account_request.html', context=context)


def password_reset_view(request):
    context = {
        'title': 'Reset Password | Assistant Analyst',
    }
    return render(request, 'gui/password_reset.html', context=context)


@login_required
def home_view(request):
    context = {
        'title': 'Home | Assistant Analyst',
        'style' : 'chat.css',
    }
    return render(request, 'gui/home.html', context=context)


@csrf_exempt
@login_required
def get_user_conversations(request):
    user = request.user
    user_conversations = Conversation.objects.filter(owner=user).order_by('-last_update')
    conversations_json = core_serializers.serialize('json', user_conversations)
    user_and_conversations = json.dumps({'username': request.user.username, 'conversations': conversations_json})
    return JsonResponse(user_and_conversations, safe=False, content_type='application/json')

@csrf_exempt
@login_required
def get_code_snippet(request, code_snippet_id: int):
    """
    Returns code snippet of the given ID.
    """
    try:
        desired_cs = CodeSnippet.objects.get(pk=code_snippet_id)
        serialized_cs = CodeSnippetSerializer(desired_cs).data
        return JsonResponse(serialized_cs, status=200)
    
    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)

@csrf_exempt
@login_required
def get_message_id_by_given_code_snippet_id(request, code_snippet_id: int):
    """
    Returns ID of message to which the code_snippet with given ID belongs
    """
    try:
        given_code_snippet = CodeSnippet.objects.get(pk=code_snippet_id)

        return JsonResponse(
            {
                'message_id': given_code_snippet.message.pk
            }, 
            status=200
        )
    
    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)


@csrf_exempt
@login_required
def make_code_snippet_compatible_with_current_df_count(request, code_snippet_id: int):
    """
    Returns the saved code snippet with code updated to work with current amount of dataframes
    """
    try:
        if request.method == 'POST':
            import re
            code_snippet = CodeSnippet.objects.get(pk=code_snippet_id)
            code_to_adjust = code_snippet.final_code

            current_dfs_fnames = agent_TBN.agent_dataframe_manager.get_dataframes_source_filenames()
            if type(current_dfs_fnames) is list:
                df_i_strs = [f'df_{i + 1}' for i in range(len(current_dfs_fnames))]
                # Ensure that all df are changed to df_1
                code_to_adjust = re.sub('df([^_])', r'df_1\g<1>', code_to_adjust) 

            else:
                df_i_strs = ['df']
                # Ensure that all df_i are changed to df
                code_to_adjust = re.sub('df_[0-9]+', 'df', code_to_adjust)
                

            # substitue all solve function related code
            sub_for = 'solve(' + ', '.join(df_i_strs) + ')'
            code_to_adjust = re.sub(r'solve\(.*?\)', sub_for, code_to_adjust)

            code_snippet.final_code = code_to_adjust
            code_snippet.save()

            serialized_cs = CodeSnippetSerializer(code_snippet).data
            return JsonResponse(serialized_cs, status=200)
        
        else:
            raise Exception('Invalid request method.')
    
    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)

    

@login_required
@csrf_exempt
def process_file_upload(request):
    response_json = {}

    conversation = _get_conversation(request)
    response_json['conversation_id'] = conversation.pk if conversation else None

    if request.method == 'POST':
        # Process the input files
        file_processing_res = check_and_process_file_input(request)  # <-- This to background

        if isinstance(file_processing_res, JsonResponse):
            response_json['file_upload_res'] = json.loads(file_processing_res.content).get('upload_response')
            return JsonResponse(response_json, status=200)
        else:
            response_json['file_upload_res'] = 'Error while processing uploaded files.'
            return JsonResponse(response_json, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    


@login_required
@csrf_exempt
def check_and_process_file_input(request):
    """
    Loads file to the agent's DataFrameManager
    """
    t_start = time.time()
    def get_file_extension(full_fname: str) -> str:
        return full_fname.split('.')[-1].lower()
    
    def get_file_name(full_fname: str) -> str:
        """
        Returns file name without file extension
        """
        return '.'.join(full_fname.split('.')[:-1])

    if request.method != 'POST':
        return JsonResponse(
            {'error': 'Invalid request method'}, 
            status=400
        )

    current_conversation = _get_conversation(request)
    # conversation_id = request.POST.get('conversation_id')
    conversation_id = current_conversation.id
    files = request.FILES.getlist('files')
    responses: list[str] = []

    if len(files) == 0:
        return JsonResponse(
            {'error': 'No files to upload'}, 
            status=400
        )
    
    # If we have files, process the files
    else: 
        # Separate dataframe files apart from annotation files 
        # while also checking for unsupported file types.
        added_dfs, added_annots = [], []
        df_files, annot_files = {}, {}
        for file in files:

            match get_file_extension(file.name):

                case 'csv' | 'xlsx' | 'parquet':
                    dict_to_add_to = df_files
                    
                case 'json':
                    dict_to_add_to = annot_files

                case _:
                    return JsonResponse({
                        'response': f'Unsupported file format {get_file_extension(file.name)} of file {file.name}. Due to this error no files were loaded into the system.', 
                        'code_snippets': None,
                        'upload_response': None,
                        'conversation_id': conversation_id,
                        'wall_time': 0.0,
                    })

            dict_to_add_to[get_file_name(file.name)] = file

        
        filepaths_to_add: list[str] = []
        annotations_to_add: list[dict | None] = []
        dfs_without_annotations: list[str] = []
        annots_used_file_names: set[str] = set()

        # Now that we have our files sorted we can load them into the Agent.
        for file_name, df_file in df_files.items():

            # Create the form for the df data
            df_form = DataTableForm(
                data={
                    'name': df_file.name, 
                    'annotation_json': None, 
                    'owner': request.user
                }, 
                files={
                    'file': df_file,
                    # 'annotation_file': annotation.file if forced_annotation is not None else None
                    'annotation_file': None
                },
            )

            df_datamodel = None
            # If supplied data are valid
            if df_form.is_valid():
                df_state = 'Supplied and Valid'
                # This created file in filesystem which Agent loads internaly
                df_datamodel = df_form.save(commit=False)
                df_datamodel.name = get_file_name(os.path.basename(df_datamodel.file.name))
                df_datamodel.save()

                current_conversation.datatables.add(df_datamodel)
                current_conversation.save()
                added_dfs.append(df_file.name)

                # Note dataframes necessary for adding them to agent
                # filepaths_to_add.append(df_datamodel.file.path)

            # If the supplied data are invalid
            else:
                df_state = 'Supplied but Invalid'
                        
            # If annotations were supplied for this df file, 
            # validate them and if they are valid use them
            try: 
                annot_form = AnnotationForm(
                    data={
                        'name': annot_files[file_name].name, 
                        'session': request.session.session_key,
                        'owner': request.user
                        }, 
                    files={
                        'file': SimpleUploadedFile(annot_files[file_name].name, annot_files[file_name].read(), content_type="application/json")
                        },
                )

                if annot_form.is_valid():
                    annotation = annot_form.save(commit=False)
                    if df_datamodel:
                        annotation_file_name = get_file_name(os.path.basename(df_datamodel.file.name)) + '.' + get_file_extension(annot_files[file_name].name)
                    else:
                        annotation_file_name = annot_files[file_name].name
                    annotation.file.save(annotation_file_name, annot_files[file_name])
                    annotation.name = get_file_name(os.path.basename(annotation.file.name))
                    annotation.save()

                    with annotation.file.file.open('r') as f:
                        forced_annotation = json.loads(f.read().decode('utf-8'))
                    
                    df_datamodel.annotation_json = json.dumps(json.dumps(forced_annotation))
                    df_datamodel.annotation_file = annotation.file
                    df_datamodel.save()
                    added_annots.append(annotation.file.name)
                                       
                    annotation_state = 'Supplied and Valid'

                    # Note annotations necessary for adding them to agent
                    annots_used_file_names.add(file_name)

                # If supplied annotations are supplied although invalid
                else:
                    forced_annotation = None
                    annotation_state = 'Supplied but Invalid'
                    dfs_without_annotations.append(df_datamodel.name)

            except:
                forced_annotation = None
                annotation_state = 'Unsupplied'
                dfs_without_annotations.append(df_datamodel.name)

            annotations_to_add.append(forced_annotation)

            responses.append(f'{file_name} was {df_state}.\nIts annotations were {annotation_state}.')

        # Note the total wall time elapsed
        wall_time_elapsed = round(time.time() - t_start, 2)
        print(f'process_file_input took {wall_time_elapsed} seconds.')
    
        return JsonResponse({
            'response': '\n\n'.join(responses),
            'code_snippets': None,
            'upload_response': 
            {
                'df_data': json.dumps(added_dfs), 
                'annot_data': json.dumps(added_annots), 
            },
            'conversation_id': conversation_id,
            'wall_time': str(wall_time_elapsed)
        }, status=200)


    
def load_conversation_data_to_agent(conversation: Conversation):
    # 1. Get all the dataframes
    # 2. Define dataframes without annotations
    # 3. Load dataframes + annotations to Agent -> missing annotations will be generated
    # 4. Add generated annotations to to df without annotations
    missing_files = []
    # ret_message = None

    datatables = conversation.datatables.all()

    # 2. Define dataframes without annotations
    dfs_without_annotations = []
    filepaths_to_add = []
    annotations_to_add = []

    for datatable in datatables:
        if datatable.file:
            file_path = datatable.file.path
            
            if not os.path.exists(file_path):
                missing_files.append(datatable.file.name)
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
            
            filepaths_to_add.append(file_path)
            
            # Check if annotation exists
            if datatable.annotation_json:
                try:
                    annotation = json.loads(datatable.annotation_json)
                    annotations_to_add.append(annotation)
                except json.JSONDecodeError:
                    print(f"Error decoding annotation JSON for DataTable {datatable.id}")
                    annotations_to_add.append(None)
                    dfs_without_annotations.append(datatable)
            else:
                annotations_to_add.append(None)
                dfs_without_annotations.append(datatable)

    # 3. Load dataframes + annotations to Agent
    if filepaths_to_add:
        agent_TBN.agent_dataframe_manager.add_data(
            filepaths_to_add, 
            forced_annotations=annotations_to_add
        )

    # 4. Add generated annotations to dfs without annotations
    for datatable in dfs_without_annotations:
        index = filepaths_to_add.index(datatable.file.path)
        generated_annotation = agent_TBN.agent_dataframe_manager.get_data_specs()[index]
        datatable.annotation_json = json.dumps(generated_annotation)
        datatable.save()

    return missing_files


@login_required
@csrf_exempt
def process_chat_input(request):
    conversation = _get_conversation(request)

    if request.method == 'POST':
        message = request.POST.get('message')  # extract User query
        # message = data.get('message')  # extract User query
        referred_msg_id = request.POST.get('reffered_msg_id')
        if referred_msg_id in ["null", "undefined", "", None]:
            referred_msg_id = None


        if message:  # if User query is present
            # Create a new user message and save it to the conversation
            message_for_answer_query = message 

            # Fetch and append referred message details to `message_for_answer_query`, if any
            if referred_msg_id:
                try:
                    referred_msg_id = int(referred_msg_id)
                    referred_message_details = get_message_with_code_snippets(conversation.id, referred_msg_id)

                    if referred_message_details:
                        message_for_answer_query += "\n\n--- Referenced Context ---\n"
                        message_for_answer_query += f"User Message: {referred_message_details['user_message']}\n"
                        message_for_answer_query += f"Bot Response: {referred_message_details['bot_response']}\n"

                        if referred_message_details['code_snippets']:
                            for i, snippet in enumerate(referred_message_details['code_snippets']):
                                message_for_answer_query += f"\nCode Snippet {i + 1}:\n{snippet['generated_code']}\n"
                except Exception as e:
                    print(f"Error fetching referred message details: {e}")

            # MV: started using user_message_obj to go with other variables better
            user_message_obj = Message.objects.create(
                author=request.user,
                parent_conversation=conversation,
                referred_to=str(referred_msg_id) if referred_msg_id else None,
                message_text=message,
                timestamp=timezone.now(),
            )
            
            # AG: Save here? MV: maybe this was placeholder for something that is done (referring messages)
            message_response = None
            try:
                # Get the assistant's response from the agent
                answer = agent_TBN.answer_query(message_for_answer_query, save_df=True) # ask the Bot

                # Create a new assistant message and save it to the conversation
                bot_message_obj = Message.objects.create(
                    author=User.objects.get(username='chatbot'),
                    parent_conversation=conversation,
                    message_text=answer[0],
                    timestamp=timezone.now(),
                )

                if answer[1] is not None and len(answer[1]) > 0:
                    for code_snippet in answer[1]:
                        # Save the code snippet to the database, associated with the bot's message
                        code_snippet_obj = CodeSnippet.objects.create(
                            message=bot_message_obj,
                            generated_code=code_snippet['code_segment'],
                            final_code=code_snippet['final_code_segment'],
                            console_out=code_snippet['traceback'],
                            plot_paths=code_snippet['plot_filenames'] if code_snippet['plot_filenames'] else '',
                            df_paths=code_snippet['path_to_saved_dfs'] if code_snippet['path_to_saved_dfs'] else '',
                            df_heads=dataframes_to_json(code_snippet['path_to_saved_dfs']),
                            query_type=code_snippet['query_type']
                        )

                # Update the conversation history summary after saving all messages and snippets
                conversation.generate_conversation_history()

                message_response = dict(MessageSerializer(bot_message_obj).data)
                # message_response['code_snippets'][0]['console_out'] = 'Testovanie'

                # Check for user-generated messages only
                user_message_count = conversation.messages.filter(author=request.user).count()
                #print(f"User message count: {user_message_count}")

                # Generate conversation name after three user messages so more context is given there
                # AG: TODO: We call this every time after 3 messages
                if user_message_count == 3:
                    conversation_messages = conversation.messages.filter(author=request.user).order_by('timestamp')[:3]
                    prompt_texts = [msg.message_text for msg in conversation_messages]

                    datatables = conversation.datatables.all()
                    if datatables.exists():
                        df = datatables[0]  
                        df_name = df.name
                        column_names = ', '.join(pd.read_csv(df.file.path).columns)

                        # Update the prompt to include dataframe and columns so it carries additional information about data apart from those three prompts
                        prompt_texts.append(f"Dataframe: {df_name}, Columns: {column_names}")

                    try:
                        # Generate the conversation name by LLM
                        conversation_name_suggestion = agent_TBN.generate_conversation_title(prompt_texts)
                        conversation.set_conversation_name(conversation_name_suggestion)
                        message_response['conversation_name'] = conversation_name_suggestion

                    except Exception as e:
                        print(f"Error while generating conversation name: {e}")
                        conversation_name_suggestion = f"General Data Exploration {timezone.now().strftime('%d-%m-%Y %H:%M')}"
                        conversation.set_conversation_name(conversation_name_suggestion)
                        message_response['conversation_name'] = conversation_name_suggestion

            except Exception as e:
                print(f"Error processing agent query: {e}")
                message_response['message_text'] = f"Error: {e}"

            conversation.last_update = timezone.now()
            conversation.save()

            # Unescaped strings in df_heads cause the problem of FE not being able to correctly decode the response
            # This is a hotfix, I believe the proper fix should be performed in the serializer 
            for serialized_code_snippet in message_response['code_snippets']:
                for df_to_dumps in serialized_code_snippet['df_heads']:
                    for dict_to_dumps in df_to_dumps['data']:
                        for k, v in dict_to_dumps.items():
                            dict_to_dumps[k] = json.dumps(v)

            return JsonResponse(message_response, status=200)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_message_with_code_snippets(conversation_id, message_id):
    """
    Serves mostly for referring to messages (any previous) in GUI in any conversation
    Retrieves a user message and bot response (message_id) within a specific conversation.
    Includes associated code snippets from the bot response.
    """
    try:
        # Fetch the user message
        user_message = Message.objects.select_related('author').get(
            id=message_id - 1,
            parent_conversation_id=conversation_id
        )

        # Fetch the bot response
        bot_response = Message.objects.select_related('author').get(
            id=message_id,
            parent_conversation_id=conversation_id,
            author__username='chatbot'
        )

        # Fetch related code snippets from the bot response
        code_snippets = list(bot_response.code_snippets.values(
            'generated_code',
            'final_code',
            'console_out',
            'plot_paths'
        ))

        # Organize and return the data
        return {
            'message_id': bot_response.id,
            'user_message': user_message.message_text,
            'bot_response': bot_response.message_text,
            'code_snippets': code_snippets,
        }

    except Message.DoesNotExist as e:
        print(f"Message not found: {e}")
        return None
        
@require_GET
def get_full_message_details(request, message_id):
    try:
        # Retrieve the parent conversation ID from the user message
        user_message = Message.objects.get(id=message_id)
        conversation_id = user_message.parent_conversation.id

        # Use the function to fetch details
        message_data = get_message_with_code_snippets(conversation_id, message_id)
        if not message_data:
            return JsonResponse({'status': 'error', 'message': 'Message not found'}, status=404)

        return JsonResponse({'status': 'success', 'message_data': message_data}, status=200)

    except Message.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Message not found'}, status=404) 

def dataframes_to_json(dfs_dir_path):
    if not dfs_dir_path or not os.path.isdir(dfs_dir_path):
        return None
    
    df_data = []
    for filename in os.listdir(dfs_dir_path):
        if filename.endswith('.csv'):  # Assuming CSV format, adjust if needed
            file_path = os.path.join(dfs_dir_path, filename)
            try:
                df = pd.read_csv(file_path)
                df_json = json.dumps(df.head(100).to_dict(orient='records'))  # Limit to first 100 rows
                df_data.append({
                    'path': file_path,
                    'data': df_json
                })
            except Exception as e:
                print(f"Error processing dataframe at {file_path}: {str(e)}")
                df_data.append({
                    'path': file_path,
                    'error': str(e)
                })
    return df_data


@login_required
@require_POST
@csrf_exempt
def create_conversation(request):
    try:
        conversation_name = f'New Conversation {timezone.now().strftime("%d.%m.%Y %H:%M")}'
        
        conversation = Conversation.objects.create(
            conversation_name=conversation_name,
            owner=request.user,
            last_update=timezone.now()
        )

        conversation_json = core_serializers.serialize('json', [conversation])[0]
        
        return JsonResponse({
            'status': 'success',
            'conversation': conversation_json,
        })
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)
    

@login_required
def load_conversation(request, conversation_id):
    try:
        agent_TBN.agent_dataframe_manager.remove_all_data()

        conversation = get_object_or_404(Conversation, id=conversation_id, owner=request.user)
        missing_files = load_conversation_data_to_agent(conversation)
        conversation_ser = ConversationSerializer(conversation)

        response_data = {
            'status': 'success',
            'conversation': conversation_ser.data,
            'missing_files': missing_files if len(missing_files)>0 else None
        }

        return JsonResponse(response_data, safe=False, content_type='application/json')
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)
    

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_conversation(request, conversation_id):
    try:
        conversation = Conversation.objects.get(id=conversation_id, owner=request.user)
        
        # Delete associated messages
        Message.objects.filter(parent_conversation=conversation).delete()
        
        # Delete associated data files
        # datatables = conversation.datatables.all()
        # for datatable in datatables:
        #     if datatable.owner:  # Has owner -> not a common table
        #         if datatable.file:
        #             if os.path.exists(datatable.file.path):
        #                 os.remove(datatable.file.path)
        #         if datatable.annotation_file:
        #             if os.path.exists(datatable.annotation_file.path):
        #                 os.remove(datatable.annotation_file.path)
        #         datatable.delete()
        
        # Delete the conversation
        conversation.delete()
        
        return JsonResponse({'status': 'success', 'message': 'Conversation deleted successfully'})
    except Conversation.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Conversation not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
@require_POST
def update_annotation(request):
    try:
        data = json.loads(request.body)
        conversation_id = data.get('conversation_id')
        annotation_data = data.get('annotation_data')

        conversation = Conversation.objects.get(id=conversation_id)
        
        # Assuming each conversation has only one DataTable
        # If multiple, you might need to adjust this logic
        datatable = conversation.datatables.filter(name=annotation_data.get('table_name')).first()
        
        if datatable:
            datatable.annotation_json = json.dumps(json.dumps(annotation_data))
            datatable.save()

            agent_TBN.agent_dataframe_manager.add_data(table_file_paths=datatable.file.path,
                                                       forced_annotations=annotation_data,
                                                       add_data_mode=AddDataMode.CHECK_RELOAD)
            
            return JsonResponse({'status': 'success', 'message': 'Annotation updated successfully'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No DataTable found for this conversation'}, status=404)

    except Conversation.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Conversation not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    

def get_agents_full_output_for_code_exec(code: str, query_type: str = 'general') -> tuple[str, list[AgentTBNCodeSnippet]]:
    """
    Returns the output which agent's answer_query would if it would generate `code` within 
    a single AgentTBNCodeSnippet with `query_type` and wouldn't debug it, just execute it.
    """
    t_start_query_answering = time.time()
    
    # Save initial code
    supplied_code = copy.deepcopy(code)

    if query_type == 'plot':
        # Count the number of plot savings
        n_plots = agent_TBN.prompt_strategy.count_savefig_calls(code)

        # Generate according amount of possivle plto names
        plot_names = [('temp/' + agent_TBN.prompt_strategy.create_possible_plot_name(agent_TBN.agent_hash)) for _ in range(n_plots)]

        # Rename the savefig calls so they save into ours plot names
        code = agent_TBN.prompt_strategy.rename_plots(code, plot_names)

    else:
        plot_names = []

    # Execute the potentialy altered code
    res, traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs = agent_TBN._code.execute_generated_code(
        code,
        tagged_query_type=query_type,
        timeout=agent_TBN.generated_code_exec_timeout,
        same_proc_exec=agent_TBN.allow_same_process_execution
    )


    if traceback == 'empty exec()':
        res = 'ERROR'
        errors = [traceback]
    else:
        errors = []
    

    code_snippet = AgentTBNCodeSnippet(
        # Query type was supplied to us
        query_type = query_type,

        # Since we do not allow any debugging to happen
        count_of_fixing_errors = 0 ,

        # Original code
        code_segment = supplied_code,

        # Potentialy altered code which got executed
        final_code_segment = code,
        
        # Since we do not allow any debugging to happen
        last_debug_prompt = '',
        
        # Whether the code executed successfully
        successful_code_execution = res != "ERROR",

        # Result state of code exectuion (e.g. 'ERROR', 'TIMEOUT', output of the code itself)
        result_of_execution = res,

        # Filenames of saved plots
        plot_filenames = plot_names, 

        # Complete paths to the saved filenames
        path_to_saved_dfs = path_to_saved_dfs,

        # Potential errors which occured during the execution of the code
        code_errors = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(errors)]),

        # Returned traceback of the execution
        traceback = traceback,
        
        # wall time code took to execute
        wall_time_final_code_runtime = t_elapsed,

        # Entire wall time the code required to get to this point
        wall_time_full = time.time() - t_start_query_answering,

        # Overall reason why the execution of code has ended
        exec_exit_reason = exec_exit_reason
    )
    
    res = agent_TBN.prompt_strategy.formulate_result([code_snippet], code)
    if res == '':

        match query_type:
            case 'general':
                res = 'Empty output from the exec() function for the text-intended answer.'

            case 'plot':
                res = code_snippet['plot_filenames']

            case _:
                pass

    return res, [code_snippet]


@csrf_exempt
def get_dfs_based_on_code(request, message_id):
    """
    Runs the code belonging to message with specified ID while returning the saved df_<n> dataframes.

    Parameters (within request)
    ---------------------------
    id
        ID of the message which CodeSnippets contain the code to be ran.
    """
    try:

        # Get the relevant DB objects
        parent_message = Message.objects.get(pk=int(message_id))
        code_snippets: list[CodeSnippet] | CodeSnippet = CodeSnippet.objects.get(message=parent_message)
        code_snippets: list[CodeSnippet] = code_snippets if type(code_snippets) is list else [code_snippets]

        # Get the code to run
        code_to_run: list[str] = []
        for code_snippet in code_snippets:
            code_to_run.append(code_snippet.final_code)

        code_to_run: str = '\n'.join(code_to_run)
        
        # Get the query type
        tagged_query_type = code_snippet.query_type

        # Run the code to get the paths to the saved dataframes
        _, _, _, _, path_to_saved_dfs_dir = agent_TBN._code.execute_generated_code(
            code_to_run,
            tagged_query_type=tagged_query_type,
            timeout=agent_TBN.generated_code_exec_timeout,
            save_df=True
        )        
        # Zip the directory
        shutil.make_archive(os.path.join(path_to_saved_dfs_dir), 'zip', path_to_saved_dfs_dir)

        # Try to Delete the directory
        try:
            shutil.rmtree(path_to_saved_dfs_dir, ignore_errors=True)
        except Exception as exp:
            print('ERROR function `get_dfs_based_on_code`: {exp}')            

        # Return the zipped directory
        return FileResponse(
            open(f'{path_to_saved_dfs_dir}.zip', 'rb'),
            as_attachment=True,
            filename='result.zip',
            content_type='application/zip'
        )
    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)


def get_agents_full_output_for_code_exec(code: str, query_type: str = 'general') -> tuple[str, list[AgentTBNCodeSnippet]]:
    """
    Returns the output which agent's answer_query would if it would generate `code` within 
    a single AgentTBNCodeSnippet with `query_type` and wouldn't debug it, just execute it.
    """
    t_start_query_answering = time.time()
    
    # Save initial code
    supplied_code = copy.deepcopy(code)

    if query_type == 'plot':
        # Count the number of plot savings
        n_plots = agent_TBN.prompt_strategy.count_savefig_calls(code)

        # Generate according amount of possivle plto names
        plot_names = [('temp/' + agent_TBN.prompt_strategy.create_possible_plot_name(agent_TBN.agent_hash)) for _ in range(n_plots)]

        # Rename the savefig calls so they save into ours plot names
        code = agent_TBN.prompt_strategy.rename_plots(code, plot_names)

    else:
        plot_names = []

    # Execute the potentialy altered code
    res, traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs = agent_TBN._code.execute_generated_code(
        code,
        tagged_query_type=query_type
    )


    if traceback == 'empty exec()':
        res = 'ERROR'
        errors = [traceback]
    else:
        errors = []
    

    code_snippet = AgentTBNCodeSnippet(
        # Query type was supplied to us
        query_type = query_type,

        # Since we do not allow any debugging to happen
        count_of_fixing_errors = 0 ,

        # Original code
        code_segment = supplied_code,

        # Potentialy altered code which got executed
        final_code_segment = code,
        
        # Since we do not allow any debugging to happen
        last_debug_prompt = '',
        
        # Whether the code executed successfully
        successful_code_execution = res != "ERROR",

        # Result state of code exectuion (e.g. 'ERROR', 'TIMEOUT', output of the code itself)
        result_of_execution = res,

        # Filenames of saved plots
        plot_filenames = plot_names, 

        # Complete paths to the saved filenames
        path_to_saved_dfs = path_to_saved_dfs,

        # Potential errors which occured during the execution of the code
        code_errors = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(errors)]),

        # Returned traceback of the execution
        traceback = traceback,
        
        # wall time code took to execute
        wall_time_final_code_runtime = t_elapsed,

        # Entire wall time the code required to get to this point
        wall_time_full = time.time() - t_start_query_answering,

        # Overall reason why the execution of code has ended
        exec_exit_reason = exec_exit_reason
    )
    
    res = agent_TBN.prompt_strategy.formulate_result([code_snippet], code)
    if res == '':

        match query_type:
            case 'general':
                res = 'Empty output from the exec() function for the text-intended answer.'

            case 'plot':
                res = code_snippet['plot_filenames']

            case _:
                pass

    return res, [code_snippet]


@csrf_exempt
def get_dfs_based_on_code(request, message_id):
    """
    Runs the code belonging to message with specified ID while returning the saved df_<n> dataframes.

    Parameters (within request)
    ---------------------------
    id
        ID of the message which CodeSnippets contain the code to be ran.
    """
    try:

        # Get the relevant DB objects
        parent_message = Message.objects.get(pk=int(message_id))
        code_snippets: list[CodeSnippet] | CodeSnippet = CodeSnippet.objects.get(message=parent_message)
        code_snippets: list[CodeSnippet] = code_snippets if type(code_snippets) is list else [code_snippets]

        # Get the code to run
        code_to_run: list[str] = []
        for code_snippet in code_snippets:
            code_to_run.append(code_snippet.final_code)

        code_to_run: str = '\n'.join(code_to_run)
        
        # Get the query type
        tagged_query_type = code_snippet.query_type

        # Run the code to get the paths to the saved dataframes
        _, _, _, _, path_to_saved_dfs_dir = agent_TBN._code.execute_generated_code(
            code_to_run,
            tagged_query_type=tagged_query_type,
            timeout=agent_TBN.generated_code_exec_timeout,
            save_df=True
        )        
        # Zip the directory
        shutil.make_archive(os.path.join(path_to_saved_dfs_dir), 'zip', path_to_saved_dfs_dir)

        # Try to Delete the directory
        try:
            shutil.rmtree(path_to_saved_dfs_dir, ignore_errors=True)
        except Exception as exp:
            print('ERROR function `get_dfs_based_on_code`: {exp}')            

        # Return the zipped directory
        return FileResponse(
            open(f'{path_to_saved_dfs_dir}.zip', 'rb'),
            as_attachment=True,
            filename='result.zip',
            content_type='application/zip'
        )

    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)





def get_agents_full_output_for_code_exec(code: str, query_type: str = 'general') -> tuple[str, list[AgentTBNCodeSnippet]]:
    """
    Returns the output which agent's answer_query would if it would generate `code` within 
    a single AgentTBNCodeSnippet with `query_type` and wouldn't debug it, just execute it.
    """
    t_start_query_answering = time.time()
    
    # Save initial code
    supplied_code = copy.deepcopy(code)

    if query_type == 'plot':
        # Count the number of plot savings
        n_plots = agent_TBN.prompt_strategy.count_savefig_calls(code)

        # Generate according amount of possivle plto names
        plot_names = [('temp/' + agent_TBN.prompt_strategy.create_possible_plot_name(agent_TBN.agent_hash)) for _ in range(n_plots)]

        # Rename the savefig calls so they save into ours plot names
        code = agent_TBN.prompt_strategy.rename_plots(code, plot_names)

    else:
        plot_names = []

    # Execute the potentialy altered code
    res, traceback, exec_exit_reason, t_elapsed, path_to_saved_dfs = agent_TBN._code.execute_generated_code(
        code,
        tagged_query_type=query_type
    )


    if traceback == 'empty exec()':
        res = 'ERROR'
        errors = [traceback]
    else:
        errors = []
    

    code_snippet = AgentTBNCodeSnippet(
        # Query type was supplied to us
        query_type = query_type,

        # Since we do not allow any debugging to happen
        count_of_fixing_errors = 0 ,

        # Original code
        code_segment = supplied_code,

        # Potentialy altered code which got executed
        final_code_segment = code,
        
        # Since we do not allow any debugging to happen
        last_debug_prompt = '',
        
        # Whether the code executed successfully
        successful_code_execution = res != "ERROR",

        # Result state of code exectuion (e.g. 'ERROR', 'TIMEOUT', output of the code itself)
        result_of_execution = res,

        # Filenames of saved plots
        plot_filenames = plot_names, 

        # Complete paths to the saved filenames
        path_to_saved_dfs = path_to_saved_dfs,

        # Potential errors which occured during the execution of the code
        code_errors = '\n'.join([f"{index}. \"{item}\"" for index, item in enumerate(errors)]),

        # Returned traceback of the execution
        traceback = traceback,
        
        # wall time code took to execute
        wall_time_final_code_runtime = t_elapsed,

        # Entire wall time the code required to get to this point
        wall_time_full = time.time() - t_start_query_answering,

        # Overall reason why the execution of code has ended
        exec_exit_reason = exec_exit_reason
    )
    
    res = agent_TBN.prompt_strategy.formulate_result([code_snippet], code)
    if res == '':

        match query_type:
            case 'general':
                res = 'Empty output from the exec() function for the text-intended answer.'

            case 'plot':
                res = code_snippet['plot_filenames']

            case _:
                pass

    return res, [code_snippet]

    

@csrf_exempt
@require_POST
def run_code(request):
    """
    Runs code for a SINGLE code snippet.

    Parameters (within request)
    ---------------------------
    edited_code: str
        String of the code to run

    code_snippet_id: int
        ID of the CodeSnippet within the DB which the edited code is considered to belong to.
        Such CodeSnippet and its Message and Conversation will be updated.
    """
    try:
        data: dict = json.loads(request.body)
        edited_code: str = data.get('edited_code', '')
        code_snippet_id: int = data.get('code_snippet_id', None)

        # Get the relevant model instances from database
        code_snippet_db = CodeSnippet.objects.get(pk=code_snippet_id)
        message_db = code_snippet_db.message
        conversation_db = message_db.parent_conversation

        # Get the agents answer
        answer_text, code_snippets = get_agents_full_output_for_code_exec(edited_code, query_type=code_snippet_db.query_type)
        
        # Extract the code snippet from the agents answer
        assert len(code_snippets) == 1, 'Multiple Code snippets generated instead of single one'
        code_snippet: CodeSnippet = code_snippets[0]

        # Updating done as suggested in Pre-Django 1.7 part of the answer
        # https://stackoverflow.com/questions/12381756/how-to-update-a-single-model-instance-retrieved-by-get-on-django-orm

        ### Update CodeSnippet
        # Here following things stay the same:
        # - message
        # - generated_code (We update only the final code)
        # - query_type
        # And following have changed
        code_snippet_db.final_code = edited_code
        code_snippet_db.console_out = code_snippet['traceback']
        code_snippet_db.plot_paths = code_snippet['plot_filenames'] if code_snippet['plot_filenames'] else ''
        code_snippet_db.df_paths = code_snippet['path_to_saved_dfs'] if code_snippet['path_to_saved_dfs'] else ''
        code_snippet_db.df_heads = dataframes_to_json(code_snippet['path_to_saved_dfs'])
        code_snippet_db.wall_time = code_snippet['wall_time_full']

        code_snippet_db.save()

        ### Now update Message
        # Here following things stay the same:
        # - author (Always bot)
        # - parent_conversation
        # - timestamp (The time the message was originaly created stays the same)
        # - upload_response SM: ???
        # And following have changed        
        message_db.message_text = answer_text
        message_db.wall_time = code_snippet['wall_time_full']

        message_db.save()

        ### Lastly update Conversation
        # Here the followin things stay the same:
        # - owner
        # - datatables
        # - conversation_name
        # And following have changed
        conversation_db.last_update = timezone.now()
        # This call updates conversation hisotry
        conversation_db.generate_conversation_history()

        conversation_db.save()

        return JsonResponse(MessageSerializer(message_db).data, status=200)

    except Exception as exp:
        return JsonResponse({'error': str(exp)}, status=400)


@require_GET
def get_dataframes(request):
    dataframes = agent_TBN.agent_dataframe_manager.get_dataframes_source_filenames()
    dataframes = [dataframes] if isinstance(dataframes, str) else dataframes
    dataframes_json = json.dumps(dataframes)
    return JsonResponse(dataframes_json, safe=False)


@require_GET
def get_common_tables(request):
    common_tables_dir = os.path.join(BASE_DIR, 'media', 'common', 'data_tables')    # AG: Check if dir exist ant start of the app 
    
    if not os.path.exists(common_tables_dir):
        print(f'Common tables directory not found on {common_tables_dir}')
        return JsonResponse({'error': f'Common tables directory not found on {common_tables_dir}'}, status=500)
    
    # Get all tables user has access to
    accessible_tables = DataTable.objects.filter(
        models.Q(all_users=True) |  # Tables available to all users
        models.Q(owner=request.user) |  # Tables owned by the user
        models.Q(access_groups__in=request.user.groups.all())  # Tables accessible via user's groups
    ).distinct()
    
    files = []
    for filename in os.listdir(common_tables_dir):
        if filename.endswith(('.csv', '.xlsx', '.json')):  # Add or remove file extensions as needed
            file_path = os.path.join(common_tables_dir, filename)
            # Only include file if user has access to corresponding DataTable
            table_name = os.path.splitext(filename)[0]
            if accessible_tables.filter(name=table_name).exists():
                file_size = os.path.getsize(file_path)
                files.append({
                    'name': filename,
                    'size': file_size,
                    'path': file_path
                })
    
    return JsonResponse(files, safe=False)


@require_GET
def get_user_tables(request):
    user_tables_dir = os.path.join(BASE_DIR, 'media', request.user.username, 'data_tables')   # AG: Check if dir exist ant start of the app 
    
    if not os.path.exists(user_tables_dir):
        print(f'User tables directory not found on {user_tables_dir}')
        return JsonResponse({'error': f'User tables directory not found on {user_tables_dir}'}, status=500)
    
    files = []
    for filename in os.listdir(user_tables_dir):
        if filename.endswith(('.csv', '.xlsx', '.json')):  # Add or remove file extensions as needed
            file_path = os.path.join(user_tables_dir, filename)
            file_size = os.path.getsize(file_path)
            files.append({
                'name': filename,
                'size': file_size,
                'path': file_path
            })
    
    return JsonResponse(files, safe=False)


@csrf_exempt
@require_POST
def update_message_reaction(request):
    data = json.loads(request.body)
    message_id = data.get('message_id')
    reaction = data.get('reaction')

    try:
        message = Message.objects.get(id=message_id)
        message.reaction = reaction
        message.save()
        return JsonResponse({'status': 'success'})
    except Message.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Message not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    

@csrf_exempt
@require_POST
def update_message_comment(request):
    data = json.loads(request.body)
    message_id = data.get('message_id')
    new_comment = data.get('comment')

    try:
        message = Message.objects.get(id=message_id)
        if message.comment:
            message.comment += f"\n\n{new_comment}"
        else:
            message.comment = new_comment
        message.save()
        return JsonResponse({'status': 'success', 'updatedComment': message.comment})
    except Message.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Message not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    

@csrf_exempt
@require_POST
def delete_user_table(request):
    try:
        data = json.loads(request.body)
        table_name = data.get('table_name').split('.')[0]
        
        # Find the table in the database
        table = DataTable.objects.filter(name=table_name, owner=request.user).first()
        
        if table:
            # Delete the file from the disk
            if table.file:
                if os.path.exists(table.file.path):
                    os.remove(table.file.path)
            
            # Delete the database entry
            table.delete()
            
            return JsonResponse({'status': 'success', 'message': f'Table {table_name} deleted successfully'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Table not found'}, status=404)
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
@require_POST
def get_dataframe_and_annotation(request):

    if request.method == 'POST':
        data = json.loads(request.body)
        df_name = data.get('df_name')
        index = data.get('index')

        # if not df_name or index is None:
        #     return JsonResponse({'error': 'Missing required parameters'}, status=400)

        dataframes = agent_TBN.agent_dataframe_manager.get_dataframes()
        if not isinstance(dataframes, list):
            dataframes = [dataframes]   # !!!

        if len(dataframes)<=0:
            return JsonResponse({'response': f'No dataframes', 
                            'df_data': None, 
                            'annot_data': None,})
    
        annotations = agent_TBN.agent_dataframe_manager.get_data_specs()
        if not isinstance(annotations, list):
            annotations = [annotations]   # !!!

        selected_df = dataframes[index]

        df_data = selected_df.head(100)
        df_json = json.dumps(df_data.to_dict(orient='records'))

        annot_data = annotations[index]
        annot_json = json.dumps(annot_data)


        return JsonResponse({'response': f'Selected dataframe: {df_name}', 
                            'df_data': df_json, 
                            'annot_data': annot_json,})
    

@csrf_exempt
@require_POST
def add_backend_dataframes_to_session(request):
    if request.method == 'POST':
        conversation = _get_conversation(request)
        tables_to_add = request.POST.get('tables_to_add', None)
        tables_to_add = json.loads(tables_to_add)

        if not tables_to_add:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        datatables = []
        for table in tables_to_add:
            if table['source'] == 'user':
                file_path = f'{request.user.username}/data_tables/{table["name"]}'
            elif table['source'] == 'common':
                file_path = f'common/data_tables/{table["name"]}'
            else:
                continue  # Skip invalid sources

            try:
                datatable = DataTable.objects.get(file=file_path)
            except DataTable.DoesNotExist:
                # Create DataTable object in the database if it doesn't exist
                datatable = DataTable.objects.create(
                    file=file_path,
                    name='.'.join(table["name"].split('.')[:-1]),
                    owner= User.objects.get(username=request.user.username) if table['source'] == 'user' else None,
                )

            datatables.append(datatable)
        
        conversation.datatables.add(*datatables)
        conversation.save()
        return JsonResponse({'status': 'success'}, status=200)
            

@csrf_exempt
@require_POST
def delete_dataframe_from_session(request):

    if request.method == 'POST':
        conversation = _get_conversation(request)
        # data = json.loads(request.body)
        df_name = request.POST.get('df_name', None)
        index = int(request.POST.get('index', None))

        if not df_name or index is None:
            return JsonResponse({'error': 'Missing required parameters'}, status=400)
        
        try:
            datatable = DataTable.objects.filter(conversations=conversation, name=df_name).first()

            if datatable:
                # Remove the DataTable from the Conversation
                conversation.datatables.remove(datatable)
                return JsonResponse({'status': 'success'}, status=200)
            else:
                return JsonResponse({'error': 'Failed to remove datatable from conversation'}, status=400)

        except Conversation.DoesNotExist:
            return JsonResponse({'error': 'Conversation not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def find_files_with_word(directory, word):
    files_with_word = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if word in file:
                files_with_word.append(os.path.join(root, file))
    return files_with_word


def image_to_base64(image_path):
    if not os.path.exists(image_path):
        # Handle the case where the image file doesn't exist
        print(f"Error: Image file '{image_path}' does not exist.")
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string

    except OSError as e:
        # Handle potential file access errors
        print(f"Error: An error occurred while reading the image file '{image_path}': {e}")
        return None

@require_GET
def get_image_base64(request):
    image_path = request.GET.get('path')
    if not image_path:
        return JsonResponse({'error': 'No image path provided'}, status=400)
    
    # Ensure the path is within your media directory to prevent unauthorized access
    # full_path = os.path.join(settings.MEDIA_ROOT, image_path)

    # if not os.path.abspath(image_path).startswith(os.path.abspath(settings.MEDIA_ROOT)):
    #     return JsonResponse({'error': 'Invalid image path'}, status=403)
    
    try:
        encoded_string = image_to_base64(image_path)
        return JsonResponse({'image': encoded_string})
    except IOError:
        return JsonResponse({'error': 'Image not found'}, status=404)



# Custom serializers:
class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username']

class CodeSnippetSerializer(ModelSerializer):
    plot_paths = SerializerMethodField()
    plot_data  = SerializerMethodField()
    df_heads   = SerializerMethodField()

    class Meta:
        model = CodeSnippet
        fields = ['id', 'generated_code', 'final_code', 'console_out', 'plot_paths', 'plot_data', 'df_paths', 'df_heads', 'wall_time', 'query_type']

    def get_plot_paths(self, obj):
        if obj.plot_paths:
            try:
                # Use ast.literal_eval to safely evaluate the string representation of the list
                return ast.literal_eval(obj.plot_paths)
            except (ValueError, SyntaxError):
                # If evaluation fails, return the original string in a list
                return [obj.plot_paths]
        return []
    
    def get_plot_data(self, obj):
        if obj.plot_paths:
            # Use ast.literal_eval to safely evaluate the string representation of the list
            paths = ast.literal_eval(obj.plot_paths)
            return [image_to_base64(path) for path in paths]
        return []
    
    def get_df_heads(self, obj):
        if obj.df_heads:
            try:
                # Step 1: Safely evaluate the outer string to a Python list
                outer_list = ast.literal_eval(obj.df_heads)

                # Ensure it's a list
                if not isinstance(outer_list, list):
                    raise ValueError("df_heads is not a list")

                # Step 2: Iterate through each dictionary in the list
                for item in outer_list:
                    if 'data' in item and isinstance(item['data'], str):
                        try:
                            # Parse the nested JSON string in the 'data' field
                            item['data'] = json.loads(item['data'])
                        except json.JSONDecodeError:
                            # Handle cases where 'data' is not valid JSON
                            # You can choose to set it to None, keep it as a string, or handle otherwise
                            item['data'] = None

                return outer_list

            except (ValueError, SyntaxError) as e:
                # Log the error if necessary
                # Return the original string or handle accordingly
                return {"error": "Invalid df_heads format", "details": str(e)}
        return []
    

class DataTableSerializer(ModelSerializer):
    class Meta:
        model = DataTable
        fields = ['id', 'name', 'file', 'annotation_file', 'annotation_json']


class MessageSerializer(ModelSerializer):
    code_snippets = CodeSnippetSerializer(many=True, read_only=True)
    message_id = serializers.IntegerField(source='id')
    author = serializers.SlugRelatedField(
        slug_field='username',
        read_only=True
    )
    
    class Meta:
        model = Message
        fields = ['message_id', 'author', 'parent_conversation', 'timestamp', 'message_text', 'upload_response', 'code_snippets', 'wall_time']


class ConversationSerializer(ModelSerializer):
    owner = UserSerializer(read_only=True)
    messages = MessageSerializer(many=True, read_only=True)
    datatables = DataTableSerializer(source='dataframes', many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'owner', 'conversation_name', 'conversation_history_summary', 'last_update', 'messages', 'datatables', ]


# def add_folder_to_datatables(folder_path, owner_username=None, is_common=True):
#     """
#     Add all files from a specified folder to the DataTable model.

#     Args:
#     folder_path (str): The path to the folder containing the files.
#     owner_username (str, optional): The username of the owner. If None and is_common is False, 
#                                     files will be added without an owner.
#     is_common (bool): If True, files will be added as common files without an owner.

#     Returns:
#     tuple: (number of files added, list of files that couldn't be added)
#     """
#     if not os.path.isdir(folder_path):
#         raise ValueError(f"The provided path is not a valid directory: {folder_path}")

#     owner = None
#     if owner_username and not is_common:
#         try:
#             owner = User.objects.get(username=owner_username)
#         except User.DoesNotExist:
#             raise ValueError(f"No user found with username: {owner_username}")

#     files_added = 0
#     failed_files = []

#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             try:
#                 # Determine if it's a data file or an annotation file
#                 is_annotation = filename.lower().endswith(('.json', '.yaml', '.yml'))
                
#                 # Create the DataTable instance
#                 datatable = DataTable(
#                     name=filename,
#                     owner=owner if not is_common else None
#                 )

#                 # Assign the file to the appropriate field
#                 if is_annotation:
#                     datatable.annotation_file.name = file_path
#                 else:
#                     datatable.file.name = file_path

#                 datatable.save()
#                 files_added += 1
#             except Exception as e:
#                 failed_files.append((filename, str(e)))

#     return files_added, failed_files