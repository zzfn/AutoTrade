const { useState, useEffect, useRef } = React;

// 根据 HTTP/HTTPS 协议返回对应的 WebSocket 协议
const getWsProtocol = () => window.location.protocol === 'https:' ? 'wss:' : 'ws:';

// 获取 WebSocket URL
const getWsUrl = (path = '/ws') => `${getWsProtocol()}//${window.location.host}${path}`;

const Card = ({ children, title, className = "" }) => (
  <div
    className={`glass rounded-2xl p-6 transition-all duration-300 ${className}`}
  >
    {title && (
      <h3 className="text-lg font-medium mb-4 text-slate-200 border-b border-slate-700/30 pb-3 flex items-center gap-2">
        {title}
      </h3>
    )}
    {children}
  </div>
);

const Button = ({
  onClick,
  children,
  variant = "primary",
  disabled = false,
  className = "",
}) => {
  const variants = {
    primary:
      "bg-cyan-500 hover:bg-cyan-400 text-slate-900 shadow-lg shadow-cyan-500/10",
    danger:
      "bg-rose-500 hover:bg-rose-400 text-white shadow-lg shadow-rose-500/10",
    secondary: "bg-slate-700 hover:bg-slate-600 text-slate-200",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

const StatusBadge = ({ status }) => {
  let colorClass = "bg-slate-500/10 text-slate-400 border-slate-500/20";
  let label = "未知";

  if (status === "running") {
    colorClass =
      "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-sm shadow-emerald-500/5";
    label = "运行中";
  }
  if (status === "stopping") {
    colorClass = "bg-amber-500/10 text-amber-400 border-amber-500/20";
    label = "停止中";
  }
  if (status === "stopped") {
    label = "已停止";
  }
  if (status === "error") {
    colorClass = "bg-rose-500/10 text-rose-400 border-rose-500/20";
    label = "错误";
  }

  return (
    <span
      className={`px-2.5 py-1 rounded-lg text-xs font-medium border ${colorClass} inline-flex items-center gap-1.5`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${status === "running" ? "bg-emerald-400 animate-pulse" : "bg-current"}`}
      ></span>
      {label}
    </span>
  );
};

// Hook for WebSocket
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("disconnected");
  const ws = useRef(null);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(url);
      ws.current.onopen = () => setStatus("connected");
      ws.current.onmessage = (e) => setData(JSON.parse(e.data));
      ws.current.onclose = () => {
        setStatus("disconnected");
        setTimeout(connect, 3000); // Reconnect
      };
    };
    connect();
    return () => ws.current?.close();
  }, [url]);

  return { data, status };
};

// 格式化时间为美东时间（Eastern Time）
// 支持的时间格式：ISO字符串、Date对象、UTC时间字符串
const formatETTime = (timestamp) => {
  if (!timestamp) return "-";

  try {
    let date;
    if (typeof timestamp === 'string') {
      // 处理 UTC 时间字符串（如 "2025-01-15T14:30:00.123456"）
      if (timestamp.includes('T')) {
        date = new Date(timestamp);
      } else {
        date = new Date(timestamp);
      }
    } else if (timestamp instanceof Date) {
      date = timestamp;
    } else {
      return "-";
    }

    // 检查日期是否有效
    if (isNaN(date.getTime())) {
      console.warn('Invalid date:', timestamp);
      return "-";
    }

    // 格式化为美东时间
    const etFormatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });

    // 获取美东时间字符串
    const etTime = etFormatter.format(date);

    // 判断是 EDT 还是 EST
    const stdFormatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      timeZoneName: 'short'
    });
    const parts = stdFormatter.formatToParts(date);
    const timeZoneName = parts.find(p => p.type === 'timeZoneName')?.value || 'ET';

    // 返回格式：2025-01-15 09:30:00 ET
    return `${etTime} ${timeZoneName}`;
  } catch (error) {
    console.error('Error formatting time:', error, timestamp);
    return "-";
  }
};

// 格式化时间为美东时间（仅时分秒）
const formatETTimeShort = (timestamp) => {
  if (!timestamp) return "-";

  try {
    let date;
    if (typeof timestamp === 'string') {
      date = new Date(timestamp);
    } else if (timestamp instanceof Date) {
      date = timestamp;
    } else {
      return "-";
    }

    if (isNaN(date.getTime())) {
      return "-";
    }

    const etFormatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });

    const etTime = etFormatter.format(date);

    const stdFormatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      timeZoneName: 'short'
    });
    const parts = stdFormatter.formatToParts(date);
    const timeZoneName = parts.find(p => p.type === 'timeZoneName')?.value || 'ET';

    return `${etTime} ${timeZoneName}`;
  } catch (error) {
    console.error('Error formatting time:', error, timestamp);
    return "-";
  }
};
