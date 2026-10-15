const bridge = window.AstrBotPluginPage;
const $ = (id) => document.getElementById(id);
let catalog = {characters: {}, styles: {}}, category = 'characters', page = 1;
const status = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
const unwrap = (value) => value?.data?.characters ? value.data : value;
function render() {
  const search = $('search').value.toLowerCase();
  const items = Object.entries(catalog[category]).filter(([name,prompt]) => `${name} ${prompt}`.toLowerCase().includes(search));
  const pages = Math.max(1, Math.ceil(items.length / 30));
  page = Math.min(page, pages);
  $('count').textContent = `${category === 'characters' ? '人物' : '风格'} ${items.length} 个`;
  $('list').replaceChildren();
  for (const [name, prompt] of items.slice((page - 1) * 30, page * 30)) {
    const row = document.createElement('article'); row.className = 'row';
    const title = document.createElement('strong'); title.textContent = name;
    const content = document.createElement('p'); content.textContent = prompt;
    const actions = document.createElement('div'); actions.className = 'actions';
    const edit = document.createElement('button'); edit.textContent = '编辑'; edit.onclick = () => openEditor(name, prompt);
    const remove = document.createElement('button'); remove.textContent = '删除';
    remove.onclick = async () => {
      if (!confirm(`删除预设「${name}」？`)) return;
      try { await bridge.apiPost('presets/delete', {kind: category, name}); await load(); }
      catch (e) { status(`删除失败：${e.message}`, true); }
    };
    actions.append(edit, remove); row.append(title, content, actions); $('list').append(row);
  }
  $('page').textContent = `${page} / ${pages}`;
  $('prev').disabled = page <= 1; $('next').disabled = page >= pages;
}
async function load() {
  try {
    catalog = unwrap(await bridge.apiGet('presets'));
    if (!catalog?.characters || !catalog?.styles) throw new Error('返回数据格式不正确');
    status(`人物 ${Object.keys(catalog.characters).length} · 风格 ${Object.keys(catalog.styles).length}。编辑后自动保存，更新插件不会覆盖。`);
    render();
  } catch(e) { status(`加载失败：${e.message}`, true); }
}
function openEditor(name = '', prompt = '') {
  $('kind').value = category; $('name').value = name; $('prompt').value = prompt; $('editor').showModal();
}
for (const kind of ['characters','styles']) $(kind).onclick = () => {
  category = kind; page = 1; for (const item of ['characters','styles']) $(item).classList.toggle('active', item === kind); render();
};
$('search').oninput = () => { page = 1; render(); };
$('prev').onclick = () => { page--; render(); };
$('next').onclick = () => { page++; render(); };
$('refresh').onclick = load; $('add').onclick = () => openEditor();
$('cancel').onclick = () => $('editor').close();
$('form').onsubmit = async (e) => {
  e.preventDefault();
  try {
    await bridge.apiPost('presets/save', {kind: $('kind').value, name: $('name').value.trim(), prompt: $('prompt').value.trim()});
    $('editor').close(); await load();
  } catch(e) { status(`保存失败：${e.message}`, true); }
};
$('export').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(catalog, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'presets.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('import').onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  try {
    if (file.size > 4 * 1024 * 1024) throw new Error('文件不能超过 4 MB');
    const data = JSON.parse(await file.text());
    if (!confirm('导入会合并到现有预设，同名项将覆盖。继续？')) return;
    await bridge.apiPost('presets/import', data); await load();
  } catch(e) { status(`导入失败：${e.message}`, true); }
  finally { e.target.value = ''; }
};
try { await bridge.ready(); await load(); }
catch(e) { status('无法连接 AstrBot，请从插件详情中的 Pages 打开本页。', true); }
