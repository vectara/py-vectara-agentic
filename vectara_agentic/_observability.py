"""
Observability for Vectara Agentic.
"""
import os
import json
from typing import Optional, Union
import pandas as pd
from .types import ObserverType
from .agent_config import AgentConfig

def setup_observer(config: AgentConfig) -> bool:
    '''
    Setup the observer.
    '''
    import phoenix as px
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.otel import register
    if config.observer == ObserverType.ARIZE_PHOENIX:
        phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", None)
        if not phoenix_endpoint:
            px.launch_app()
            tracer_provider = register(endpoint='http://localhost:6006/v1/traces', project_name="vectara-agentic")
        elif 'app.phoenix.arize.com' in phoenix_endpoint:   # hosted on Arize
            phoenix_api_key = os.getenv("PHOENIX_API_KEY", None)
            if not phoenix_api_key:
                raise ValueError("Arize Phoenix API key not set. Please set PHOENIX_API_KEY environment variable.")
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
            tracer_provider = register(endpoint=phoenix_endpoint, project_name="vectara-agentic")
        else:       # Self hosted Phoenix
            tracer_provider = register(endpoint=phoenix_endpoint, project_name="vectara-agentic")
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        return True
    print("No observer set.")
    return False


def _extract_fcs_value(output: Union[str, dict]) -> Optional[float]:
    '''
    Extract the FCS value from the output.
    '''
    try:
        output_json = json.loads(output)
        if 'metadata' in output_json and 'fcs' in output_json['metadata']:
            return output_json['metadata']['fcs']
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {output}")
    except KeyError:
        print(f"'fcs' not found in: {output_json}")
    return None


def _find_top_level_parent_id(row: pd.Series, all_spans: pd.DataFrame) -> Optional[str]:
    '''
    Find the top level parent id for the given span.
    '''
    current_id = row['parent_id']
    while current_id is not None:
        parent_row = all_spans[all_spans.index == current_id]
        if parent_row.empty:
            break
        new_parent_id = parent_row['parent_id'].iloc[0]
        if new_parent_id == current_id:
            break
        if new_parent_id is None:
            return current_id
        current_id = new_parent_id
    return current_id


def eval_fcs() -> None:
    '''
    Evaluate the FCS score for the VectaraQueryEngine._query span.
    '''
    import phoenix as px
    from phoenix.trace.dsl import SpanQuery
    from phoenix.trace import SpanEvaluations
    query = SpanQuery().select(
        "output.value",
        "parent_id",
        "name"
    )
    client = px.Client()
    all_spans = client.query_spans(query, project_name="vectara-agentic")
    vectara_spans = all_spans[all_spans['name'] == 'VectaraQueryEngine._query'].copy()
    vectara_spans['top_level_parent_id'] = vectara_spans.apply(
        lambda row: _find_top_level_parent_id(row, all_spans), axis=1
    )
    vectara_spans['score'] = vectara_spans['output.value'].apply(_extract_fcs_value)

    vectara_spans.reset_index(inplace=True)
    top_level_spans = vectara_spans.copy()
    top_level_spans['context.span_id'] = top_level_spans['top_level_parent_id']
    vectara_spans = pd.concat([vectara_spans, top_level_spans], ignore_index=True)
    vectara_spans.set_index('context.span_id', inplace=True)

    px.Client().log_evaluations(
        SpanEvaluations(
            dataframe=vectara_spans,
            eval_name="Vectara FCS",
        ),
    )
