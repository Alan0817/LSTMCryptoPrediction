import json
from tools.registry import ToolRegistry
from web_search import WebSearchProvider,WebSearchResult,search_web
from retrieval.planning import web_plan_metrics
class Fake(WebSearchProvider):
 def __init__(self,rows=(),error=None): self.rows=rows; self.error=error; self.calls=[]
 def search(self,q,max_results=5):
  self.calls.append((q,max_results))
  if self.error: raise self.error
  return self.rows
def test_web_tool_registry_and_provenance():
 p=Fake([WebSearchResult('Title','https://example.com','Snippet','Example',None)])
 r=ToolRegistry(web_search_provider=p); out=r.execute('search_web',{'query':'NVIDIA latest','max_results':1})
 assert 'search_web' in [x.name for x in r.list_tools()] and out['results'][0]['url']=='https://example.com' and json.dumps(out)
 assert 'search_web' not in [x.name for x in ToolRegistry().list_tools()]
def test_web_statuses_and_planning():
 assert search_web('q')['status']=='unavailable'
 assert search_web('q',provider=Fake(error=RuntimeError('secret path')))['status']=='search_error'
 t=[{'event':'tool_requested','name':'search_web','arguments':{'query':'a'}},{'event':'tool_requested','name':'search_web','arguments':{'query':'b'}}]
 assert web_plan_metrics(t)['materially_changed_query']
