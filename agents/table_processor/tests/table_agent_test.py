import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from table_processor import TableAgent
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()
api_key = os.environ.get("OPENAI_TOKEN")

if __name__ == '__main__':
    # import os
    project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    table_file_path = os.path.join(project_root, 'data', 'files_processed', 'table_data', 'tables', 'msg_alternators.csv')
    data_specs_path = os.path.join(project_root, 'data', 'files_processed', 'table_data', 'annotations')
    tmp_path = os.path.join(project_root, 'data', 'files_processed', 'table_data', 'temp')

    # table_agent = TableAgent(table_file_path=table_file_path,
    #                          data_specs_dir_path=data_specs_path)

    # resp = table_agent.answer_query('OEM number for Audi A6 2018 with 3.0 TFSI motor')
    # print(resp[0])

    # Verify paths exist
    if not os.path.exists(data_specs_path):
        print(f"Data specifications path does not exist: {data_specs_path}")
        os.makedirs(data_specs_path, exist_ok=True)
        print(f"Created specifications directory: {data_specs_path}")
    if not os.path.exists(tmp_path):
        print(f"Temp path does not exist: {tmp_path}")
        os.makedirs(tmp_path, exist_ok=True)
        print(f"Created temp directory: {tmp_path}")
    
    client = OpenAI(api_key=api_key, timeout=30, max_retries=3)

    table_agent = TableAgent(
        client=client,
        prompt_strategy='hybrid_code_text',
        table_file_path=table_file_path,
        data_specs_dir_path=data_specs_path,
        tmp_file_path=tmp_path,
        generated_code_exec_timeout=60
    )

    # resp, code = table_agent.answer_query('OEM number for Audi A6 2018 with 3.0 TFSI motor')
    # resp, code = table_agent.answer_query('Which cable do I need for generator with oem 04L903018?')
    # resp, code = table_agent.answer_query('Which protocol do generator from audi A3 use?')
    # resp, code = table_agent.answer_query('Which protocol do generator from audi A3 use?')
    # resp, code = table_agent.answer_query('Can I get live data for Audi Q5 2014')
    resp, code = table_agent.answer_query('which alternators work with 48V?')

    print(resp)