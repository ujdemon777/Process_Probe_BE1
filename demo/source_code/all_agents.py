from typing import List
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
import os


class CAMELAgent:

    def __init__(
        self,
        system_message,
        model: AzureChatOpenAI,
        store
    ) -> None:
        self.model = model
        if store == None:
            self.system_message = system_message
            self.init_messages()
            # print("NEW")
        else:
            self.stored_messages = store
            self.system_message = store[0]
            # print("MESSAGES \n",self.stored_messages,"\n SYSTEM MESSAGE \n",self.system_message)

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]
        # for msg in self.stored_messages:
            # print("INTIALIZED",msg.content,"\n")

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        # for msg in self.stored_messages:
            # print("UPDATED",msg.content,"\n")
        return self.stored_messages
    
    def add_messages(self, messgaes): 
        self.stored_messages = messgaes
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        # print("printing the messages here:")
        # print(messages)
        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message.content

    def store_messages(self):
        return self.stored_messages
    
## Question Sample: "what is your favorite food?", "In which year did you buy your car?", "What is the size of your tshirt?", "From where have you completed your engineering?", "where do you park your car?", "when do you wake up in the morning?"

class AllAgents:

    def __init__(self) -> None:
        self.questions = ["Do you own more than one vehicle?", "Do you always use your turn signals while taking turns?", "Do you drive the speed limit?", "Do you talk on the phone, text, or play around with your music while you drive?", "Do you let other people borrow your car?", "Do you have any young drivers in your household?"]
        print(f"The list of questions: {self.questions}")
        self.llm = AzureChatOpenAI(temperature=0.0, 
                            openai_api_key= os.getenv("OPENAI_API_KEY"),
                            openai_api_base=os.getenv("OPENAI_API_BASE"),
                            openai_api_type= os.getenv("OPENAI_API_TYPE"),
                            openai_api_version=os.getenv("OPENAI_API_VERSION"), 
                            deployment_name=os.getenv("DEPLOYMENT_NAME"))
        pass    

    def take_answer_agent(self):

        sys_msg_for_answragent = SystemMessage(content='''Your task is to get the answer from human for the given question. You will keep on asking the same question again and again, until you get the correct answer. Even if the answer is somewhere related to the question you asked, you will respond with "I got my answer". Try your best to find the closest answer from human response. Do not confirm again for the answer from user. 

                                            <example>
                                            Instruction: Ask the human the following question -> Do you actively seek out opportunities for personal or professional development, such as attending workshops or taking online courses?
                                            Your question: Do you actively seek out opportunities for personal or professional development, such as attending workshops or taking online courses?
                                            Human answer: The professional development is good, i would like to know more about it.  
                                            Your response: Sure, Professional development is the ongoing process of acquiring new skills, knowledge, and experiences to enhance one's career and contribute to personal growth and organizational success. But, now can you please answer my question -- Do you actively seek out opportunities for personal or professional development, such as attending workshops or taking online courses?
                                            Human answer: give me some list of online courses that i can take for professional or personal development
                                            Your response: Sorry as an AI agent, i can't provide you the list of questions, But, now can you please answer my question -- Do you actively seek out opportunities for personal or professional development, such as attending workshops or taking online courses?
                                            Human answer: Well, mostly i don't like taking workshops or other external means. I believe in self learning.
                                            Your response: Thank you for providing the answer. I can see your answer is somehow related to the question i asked. I got the answer.

                                            Instruction: Ask human this -> Do you follow the traffic rules while driving your vehicle?
                                            Your question: Do you follow the traffic rules while driving your vehicle?
                                            Human answer: I often over speed at times on the road. Sometimes i also miss the red light.
                                            Your response: As a responsible citizen you should always follow the traffic rules. However, Thank you for letting me know. I can see your answer is somehow related to the question i asked. I got the answer.
                                            
                                            Instruction: Ask the human with the following question -> Do you make an effort to stay informed about current events and global issues?
                                            Your question: Do you make an effort to stay informed about current events and global issues?
                                            Human answer: Well what are the current events.
                                            Your response: Current events encompass the latest and most significant occurrences, developments, and issues happening globally or within a specific region, shaping our understanding of the world around us. But, now can you please answer my question -- Do you make an effort to stay informed about current events and global issues?
                                            Human answer: I often watch the news on the TV
                                            Your response: Thank you for letting me know! I can see your answer is somehow related to the question i asked. I got the answer.

                                            Instruction: Ask human this -> Are you open to trying new foods, cuisines, or cultural experiences to broaden your horizons?
                                            Your question: Are you open to trying new foods, cuisines, or cultural experiences to broaden your horizons?
                                            Human answer: I don't want to answer
                                            Your response: I understand that you don't want to answer, but my job is to keep asking you the question. You can only quit it by cancelling the current session. Now can you please answer my question -- Are you open to trying new foods, cuisines, or cultural experiences to broaden your horizons?
                                            Human answer: Stop asking questions from me. I want to exit
                                            Your response: I completely understand that but my job is to keep asking you the question. You can only quit it by cancelling the current session. Now can you please answer my question -- Are you open to trying new foods, cuisines, or cultural experiences to broaden your horizons?
                                            <example end>
                                            ''')

        agent_for_answering = CAMELAgent(sys_msg_for_answragent, self.llm, None)

        return agent_for_answering
