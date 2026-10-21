import React from 'react'
import ReactDOM from 'react-dom/client'
import TraceViewerPage from './trace/TraceViewerPage'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><TraceViewerPage /></React.StrictMode>,
)
