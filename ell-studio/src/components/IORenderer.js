import React from 'react';

const typeMatchers = {
  ToolResult: (data) => data && typeof data === 'object' && 'tool_call_id' in data && 'result' in data,
  ToolCall: (data) => data && typeof data === 'object' && 'tool' in data && 'params' in data,
  ContentBlock: (data) => data && typeof data === 'object' && ['text', 'image', 'audio', 'tool_call', 'parsed', 'tool_result'].some(key => key in data),
  Message: (data) => data && typeof data === 'object' && 'role' in data && 'content' in data,
};

const preprocessData = (data) => {
  if (typeMatchers.Message(data)) {
    const { role, content } = data;
    if (content.length === 1) {
      const block = content[0];
      if (block.text) {
        return { type: 'Message', role, content: block.text };
      }
      if (block.tool_call) {
        return { type: 'Message', role, content: preprocessData(block.tool_call) };
      }
      const complexType = ['image', 'audio', 'parsed', 'tool_result'].find(type => block[type]);
      if (complexType) {
        return { type: 'Message', role, content: preprocessData(block[complexType]) };
      }
    }
    return { type: 'Message', role, content: content.map(preprocessData) };
  }
  
  if (typeMatchers.ContentBlock(data)) {
    const contentType = Object.keys(data).find(key => ['text', 'image', 'audio', 'tool_call', 'parsed', 'tool_result'].includes(key));
    return { type: 'ContentBlock', content: preprocessData(data[contentType]) };
  }
  
  if (Array.isArray(data)) {
    return data.map(preprocessData);
  }
  
  if (typeof data === 'object' && data !== null) {
    const typeRenderer = Object.entries(typeMatchers).find(([, matcher]) => matcher(data));
    if (typeRenderer) {
      const [type] = typeRenderer;
      return { type, ...data };
    }
    return Object.fromEntries(Object.entries(data).map(([key, value]) => [key, preprocessData(value)]));
  }
  
  return data;
};

const renderInline = (data, customRenderers) => {
  if (typeof data === 'object' && data !== null && 'type' in data) {
    const { type, ...rest } = data;
    const typeColor = {
      ToolResult: 'blue',
      ToolCall: 'purple',
      ContentBlock: 'orange',
      Message: 'teal'
    }[type];
    
    return (
      <span className="text-gray-300">
        <span className={`text-${typeColor}-400`}>{type}</span>(
        {type === 'Message' ? (
          <>
            {renderInline(rest.content, customRenderers)},
            <span className="text-sky-300">role</span>=<span className="text-green-300">"{rest.role}"</span>
          </>
        ) : (
          Object.entries(rest).map(([key, value], index, arr) => (
            <React.Fragment key={key}>
              <span className="text-sky-300">{key}</span>=
              {renderInline(value, customRenderers)}
              {index < arr.length - 1 && ', '}
            </React.Fragment>
          ))
        )}
        )
      </span>
    );
  }
  
  if (Array.isArray(data)) {
    return (
      <span className="text-gray-300">
        [{data.map((item, index) => (
          <React.Fragment key={index}>
            {index > 0 && ', '}
            {renderInline(item, customRenderers)}
          </React.Fragment>
        ))}]
      </span>
    );
  }
  
  if (typeof data === 'object' && data !== null) {
    return (
      <span className="text-gray-300">
        {'{'}
        {Object.entries(data).map(([key, value], index, arr) => (
          <React.Fragment key={key}>
            <span className="text-sky-300">{key}</span>: {renderInline(value, customRenderers)}
            {index < arr.length - 1 && ', '}
          </React.Fragment>
        ))}
        {'}'}
      </span>
    );
  }
  
  if (typeof data === 'string') {
    return <span className="text-green-300">"{data}"</span>;
  }
  
  return <span className="text-yellow-300">{JSON.stringify(data)}</span>;
};

const renderNonInline = (data, customRenderers, level = 0, isArrayItem = false, postfix = '') => {
  if (typeof data === 'object' && data !== null && 'type' in data) {
    const { type, ...rest } = data;
    const typeColor = {
      ToolResult: 'blue',
      ToolCall: 'purple',
      ContentBlock: 'orange',
      Message: 'teal'
    }[type];
    
    return (
      <div>
        <span className={`text-${typeColor}-400`}>{type}</span>(
        {type === 'Message' ? (
          <>
            <Indent level={level + 1}>
              {renderNonInline(rest.content, customRenderers, level + 1, false, ',')}
            </Indent>
            <Indent level={level + 1}>
              <span className="text-sky-300">role</span>: <span className="text-green-300">"{rest.role}"</span>
            </Indent>
          </>
        ) : (
          Object.entries(rest).map(([key, value], index, arr) => (
            <React.Fragment key={key}>
              <Indent level={level + 1}>
                <span className="text-sky-300">{key}</span>: {
                  renderNonInline(value, customRenderers, level + 1, false, index < arr.length - 1 ? ',' : '')
                }
              </Indent>
            </React.Fragment>
          ))
        )}
        <div>){postfix}</div>
      </div>
    );
  }
  
  if (Array.isArray(data)) {
    return (
      <>
        {isArrayItem ? '[ ' : '['}
        {data.map((item, index) => (
          <React.Fragment key={index}>
            <Indent level={level + 1}>
              {renderNonInline(item, customRenderers, level + 1, true, index < data.length - 1 ? ',' : '')}
            </Indent>
          </React.Fragment>
        ))}
        <div>{isArrayItem ? ' ]' : ']'}{postfix}</div>
      </>
    );
  }
  
  if (typeof data === 'object' && data !== null) {
    return (
      <>
        {'{'}
        {Object.entries(data).map(([key, value], index, arr) => (
          <React.Fragment key={key}>
            <Indent level={level + 1}>
              <span className="text-sky-300">{key}</span>: {
                renderNonInline(value, customRenderers, level + 1, false, index < arr.length - 1 ? ',' : '')
              }
            </Indent>
          </React.Fragment>
        ))}
       <div>{'}'}{postfix}</div>
      </>
    );
  }
  
  if (typeof data === 'string') {
    if (data.includes('\n')) {
      const lines = data.split('\n');
      return (
        <div>
          <span className="text-green-300">"""</span>
          {lines.map((line, index) => (
            <div>
              <span className="text-green-300">{line}</span>
            </div>
          ))}
        <span className="text-green-300">"""</span>{postfix}
        </div>
      );
    } else {
      return (
        <span>
          <span className="text-green-300">"{data}"</span>{postfix}
        </span>
      );
    }
  }
  
  return (
    <span>
      {renderInline(data, customRenderers)}
      {postfix}
    </span>
  );
};

const Indent = ({ level, children }) => (
  <div style={{ marginLeft: `${level * 7}px` }}>
    <span>{children}</span>
  </div>
);

const IORenderer = ({ content, customRenderers = [], inline = true }) => {
  try {
    const parsedContent = JSON.parse(content);
    const preprocessedContent = preprocessData(parsedContent);
    return (
      <div className="text-sm font-mono">
        {inline ? renderInline(preprocessedContent, customRenderers) : renderNonInline(preprocessedContent, customRenderers)}
      </div>
    );
  } catch {
    return <span className="text-sm text-gray-300">{content}</span>;
  }
};

export default IORenderer;