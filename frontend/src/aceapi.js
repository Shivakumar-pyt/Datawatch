import React, { useState } from 'react';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-mysql';
import 'ace-builds/src-noconflict/theme-github';

function MyAceEditor() {
  const [query, setQuery] = useState('');
  const [syntaxError, setSyntaxError] = useState('');

  const handleQueryChange = (newQuery) => {
    setQuery(newQuery);
  };

  const handleCheckSyntax = () => {
    // Validate MySQL syntax
    try {
      window.ace.require('ace/mode/mysql').parse(query);
      setSyntaxError('');
    } catch (error) {
      setSyntaxError(error.message);
    }
  };

  return (
    <div>
      <AceEditor
        mode="mysql"
        theme="github"
        onChange={handleQueryChange}
        value={query}
        name="query-editor"
        editorProps={{ $blockScrolling: true }}
      />
      <button onClick={handleCheckSyntax}>Check Syntax</button>
      {syntaxError && <div>Error: {syntaxError}</div>}
    </div>
  );
}

export default MyAceEditor;
