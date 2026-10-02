// Разбор SSE вручную: fetch+POST нельзя отдать EventSource, он умеет только GET.
// Вынесено из шаблона, чтобы тесты гоняли тот же код, что и браузер
// (tests/test_api.py, через node).

// sse-starlette завершает строки «\r\n», и разделитель событий — «\r\n\r\n».
// Поиск одного «\n\n» не находил ни одного события: страница навсегда зависала
// на «Ищу в документах…». Спецификация SSE допускает \r\n, \r и \n.
// Одиночный \r в конце куска может оказаться половиной \r\n — его не трогаем.
function normalizeNewlines(buffer) {
  return buffer.replace(/\r\n|\r(?!$)/g, "\n");
}

// Отделить готовые события от недочитанного хвоста.
function takeEvents(buffer) {
  buffer = normalizeNewlines(buffer);
  const events = [];
  let split;
  while ((split = buffer.indexOf("\n\n")) !== -1) {
    events.push(parseEvent(buffer.slice(0, split)));
    buffer = buffer.slice(split + 2);
  }
  return { events, rest: buffer };
}

function parseEvent(raw) {
  let name = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  let data = {};
  try { data = JSON.parse(dataLines.join("\n")); } catch (_) {}
  return { name, data };
}

async function readStream(reader, onEvent) {
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const { events, rest } = takeEvents(buffer + decoder.decode(value, { stream: true }));
    buffer = rest;
    events.forEach(onEvent);
  }
}

if (typeof module !== "undefined") module.exports = { takeEvents, parseEvent };
