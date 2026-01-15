from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    msgs = [generate_chain.invoke({"messages": state["messages"]})]
    print("*" * 40)
    print("generation_node: ", msgs)
    return {"messages": msgs}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    print("*" * 40)
    print("reflection_node: ", res.content)
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()



if __name__ == "__main__":
    print("Hello Reflection Agent!")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                        @LangChainAI
                — newly Tool Calling feature is seriously underrated.

                After a long wait, it's  here- making the implementation of agents across different models with 
                function calling - super easy.

                Made a video covering their newest blog post

                """
            )
        ]
    }
    result = graph.invoke(inputs)
    print("====" * 40)
    print("res", result)
    print("====" * 40)
    print("content: ", result["messages"][-1].content)
