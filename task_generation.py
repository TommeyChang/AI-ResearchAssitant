# ========= All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= All Rights Reserved. =========
import multiprocessing
import os

from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.generators import (
    AISocietyTaskPromptGenerator,
    SystemMessageGenerator,
)
from camel.messages import BaseMessage
from camel.types import RoleType, TaskType, ModelPlatformType, ModelType
from camel.models import ModelFactory

load_dotenv()


def generate_tasks(
    role_names: str,
    task_generator_prompt: str,
    start_idx: int = 1,
    num_tasks: int = 10,
    model=None,
) -> None:
    sys_msg_generator = SystemMessageGenerator(task_type=TaskType.AI_SOCIETY)

    assistant_sys_msg = sys_msg_generator.from_dict(
        dict(), role_tuple=("Task Generator", RoleType.DEFAULT)
    )
    assistant_agent = ChatAgent(assistant_sys_msg, model=model)

    user_msg = BaseMessage.make_user_message(
        role_name="Task Generator", content=task_generator_prompt
    )

    assistant_response = assistant_agent.step(user_msg)

    if assistant_response.terminated or len(assistant_response.msgs) == 0:
        raise RuntimeError("Assistant agent terminated unexpectedly.")

    tasks = assistant_response.msg.content.split("\n")

    filted_tasks = []
    for task in tasks:
        task = task.strip()
        if len(tasks) <= 5:
            continue
        if str(start_idx) + "." in task:
            start_idx += 1
            filted_tasks.append(task)
        if start_idx > num_tasks:
            break

    with open(f"community/tasks/{'_'.join(role_names)}.txt", "w") as file:
        file.write("\n".join(tasks))


def main(model=None) -> None:
    num_tasks = 6
    start_idx = 1

    task_generator_prompt_generator = AISocietyTaskPromptGenerator(
        num_tasks=num_tasks
    ).from_role_files(
        user_role_names_path="./community/user_list.txt",
        assistant_role_names_path="./community/assitant_list.txt",
    )

    pool = multiprocessing.Pool()

    for task_generator_prompt, role_names in task_generator_prompt_generator:
        if not os.path.exists(f"community/tasks/{'_'.join(role_names)}.txt"):
            print(f"Generating tasks for {role_names}")
            pool.apply_async(
                generate_tasks,
                role_names,
                task_generator_prompt,
                start_idx,
                num_tasks,
                model,
            )

    pool.close()
    pool.join()


if __name__ == "__main__":
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK, model_type=ModelType.DEEPSEEK_CHAT
    )

    main(model)
