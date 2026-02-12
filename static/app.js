// Utility function to populate location select dropdowns in forms

async function fetchLookup(fieldKey) {
  const res = await fetch(`/api/lookups/${encodeURIComponent(fieldKey)}`);
  if (!res.ok) return [];
  return await res.json();
}

async function populateLookupSelect(fieldKey, selectId, selectedValue = "") {
  const el = document.getElementById(selectId);
  if (!el) return;

  const items = await fetchLookup(fieldKey);

  if (!items.length) {
    el.innerHTML = `<option value="">(No values set - add in Field Management)</option>`;
    // 🔥 ensure searchable wrappers / listeners sync even on empty
    el.value = "";
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return;
  }

  el.innerHTML =
    `<option value=""></option>` +
    items.map(it => {
      const sel = (String(it.code) === String(selectedValue)) ? "selected" : "";
      return `<option value="${it.code}" ${sel}>${it.label}</option>`;
    }).join("");

  // If existing value isn't in lookup list (legacy), append it so nothing breaks
  if (selectedValue && !items.some(it => String(it.code) === String(selectedValue))) {
    const opt = document.createElement("option");
    opt.value = selectedValue;
    opt.textContent = `${selectedValue} (legacy)`;
    opt.selected = true;
    el.appendChild(opt);
  }

  // 🔥 critical: programmatic selection must notify searchable wrapper
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

async function populateLookupFilter(fieldKey, selectId, selectedValue = "") {
  const el = document.getElementById(selectId);
  if (!el) return;

  const items = await fetchLookup(fieldKey);

  // blank option = no filter
  el.innerHTML =
    `<option class="filter-option" value=""></option>` +
    items.map(it => `<option class="filter-option" value="${it.code}">${it.label}</option>`).join("");

  if (selectedValue) el.value = selectedValue;

  // keep any wrappers in sync
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

// Searchable drop down vanilla js

function makeSelectSearchable(selectEl, { placeholder = "Type to search...", allowClear = true } = {}) {
  if (!selectEl || selectEl.dataset.searchableInit) return;
  selectEl.dataset.searchableInit = "1";

  // Ensure there is a blank option if allowClear is enabled
  if (allowClear) {
    const hasBlank = Array.from(selectEl.options).some(o => o.value === "");
    if (!hasBlank) {
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "";
      selectEl.insertBefore(blank, selectEl.firstChild);
    }
  }

  // Hide original select, keep it for real value submission
  selectEl.style.display = "none";

  const wrap = document.createElement("div");
  wrap.className = "ss-wrap";
  selectEl.parentNode.insertBefore(wrap, selectEl);
  wrap.appendChild(selectEl);

  const input = document.createElement("input");
  input.className = "ss-input";
  input.placeholder = placeholder;
  wrap.insertBefore(input, selectEl);

  const list = document.createElement("div");
  list.className = "ss-list";
  wrap.appendChild(list);

  function selectedOption() {
    return selectEl.options[selectEl.selectedIndex] || null;
  }
  function selectedLabel() {
    const opt = selectedOption();
    return opt ? opt.textContent : "";
  }

  function open() { list.style.display = "block"; }
  function close() { list.style.display = "none"; }

  function render(filterText = "") {
    const ft = (filterText || "").toLowerCase();
    list.innerHTML = "";

    const opts = Array.from(selectEl.options);

    // Include blank option if allowClear
    const visible = opts.filter(o => {
      if (o.value === "") return allowClear; // blank
      return o.textContent.toLowerCase().includes(ft);
    });

    if (!visible.length) {
      const empty = document.createElement("div");
      empty.className = "ss-empty";
      empty.textContent = "No matches";
      list.appendChild(empty);
      return;
    }

    for (const o of visible) {
      const item = document.createElement("div");
      item.className = "ss-item";
      item.textContent = (o.value === "" ? "— Clear —" : o.textContent);
      item.onclick = () => {
        selectEl.value = o.value;
        // 🔥 bubble so any listeners higher up can respond
        selectEl.dispatchEvent(new Event("change", { bubbles: true }));
        input.value = (o.value === "" ? "" : o.textContent);
        close();
      };
      list.appendChild(item);
    }
  }

  // Initialize input to current selection
  input.value = selectedOption() && selectedOption().value !== "" ? selectedLabel() : "";

  input.addEventListener("focus", () => { render(input.value); open(); });
  input.addEventListener("input", () => { render(input.value); open(); });

  // On blur, snap back to current selected label (prevents free-text "fake" values)
  input.addEventListener("blur", () => {
    setTimeout(() => {
      input.value = (selectEl.value === "" ? "" : selectedLabel());
      close();
    }, 150);
  });

  // Sync input if select changes programmatically
  selectEl.addEventListener("change", () => {
    input.value = (selectEl.value === "" ? "" : selectedLabel());
  });

  // 🔥 Initial sync in case options/value were set before wrapper existed
  selectEl.dispatchEvent(new Event("change", { bubbles: true }));
}

function initSearchableSelects(root = document) {
  root.querySelectorAll("select.searchable").forEach(sel =>
    makeSelectSearchable(sel, { allowClear: true })
  );
}


const lb = document.getElementById("logoutBtn");
if (lb) {
  lb.onclick = async () => {
    await fetch("/api/logout", { method:"POST" });
    window.location.href = "/shop/inventory";
  };
}




// // Utility function to populate location select dropdowns in forms

// async function fetchLookup(fieldKey) {
//   const res = await fetch(`/api/lookups/${encodeURIComponent(fieldKey)}`);
//   if (!res.ok) return [];
//   return await res.json();
// }

// async function populateLookupSelect(fieldKey, selectId, selectedValue = "") {
//   const el = document.getElementById(selectId);
//   if (!el) return;

//   const items = await fetchLookup(fieldKey);

//   if (!items.length) {
//   el.innerHTML = `<option value="">(No values set - add in Field Management)</option>`;
//   return;
//   }

//   el.innerHTML =`<option value=""></option>` + items.map(it => {
//     const sel = (String(it.code) === String(selectedValue)) ? "selected" : "";
//     return `<option value="${it.code}" ${sel}>${it.label}</option>`;
//   }).join("");

//   // If existing value isn't in lookup list (legacy), append it so nothing breaks
//   if (selectedValue && !items.some(it => String(it.code) === String(selectedValue))) {
//     const opt = document.createElement("option");
//     opt.value = selectedValue;
//     opt.textContent = `${selectedValue} (legacy)`;
//     opt.selected = true;
//     el.appendChild(opt);
//   } 
// }


// async function populateLookupFilter(fieldKey, selectId, selectedValue = "") {
//   const el = document.getElementById(selectId);
//   if (!el) return;

//   const items = await fetchLookup(fieldKey);

//   // blank option = no filter
//   el.innerHTML = `<option value=""></option>` + items.map(it => {
//     return `<option value="${it.code}">${it.label}</option>`;
//   }).join("");

//   if (selectedValue) el.value = selectedValue;
// }

// // Searchable drop down vanilla js

// function makeSelectSearchable(selectEl, { placeholder = "Type to search...", allowClear = true } = {}) {
//   if (!selectEl || selectEl.dataset.searchableInit) return;
//   selectEl.dataset.searchableInit = "1";

//   // Ensure there is a blank option if allowClear is enabled
//   if (allowClear) {
//     const hasBlank = Array.from(selectEl.options).some(o => o.value === "");
//     if (!hasBlank) {
//       const blank = document.createElement("option");
//       blank.value = "";
//       blank.textContent = "";
//       selectEl.insertBefore(blank, selectEl.firstChild);
//     }
//   }

//   // Hide original select, keep it for real value submission
//   selectEl.style.display = "none";

//   const wrap = document.createElement("div");
//   wrap.className = "ss-wrap";
//   selectEl.parentNode.insertBefore(wrap, selectEl);
//   wrap.appendChild(selectEl);

//   const input = document.createElement("input");
//   input.className = "ss-input";
//   input.placeholder = placeholder;
//   wrap.insertBefore(input, selectEl);

//   const list = document.createElement("div");
//   list.className = "ss-list";
//   wrap.appendChild(list);

//   function selectedOption() {
//     return selectEl.options[selectEl.selectedIndex] || null;
//   }
//   function selectedLabel() {
//     const opt = selectedOption();
//     return opt ? opt.textContent : "";
//   }

//   function open() { list.style.display = "block"; }
//   function close() { list.style.display = "none"; }

//   function render(filterText = "") {
//     const ft = (filterText || "").toLowerCase();
//     list.innerHTML = "";

//     const opts = Array.from(selectEl.options);

//     // Include blank option if allowClear
//     const visible = opts.filter(o => {
//       if (o.value === "") return allowClear; // blank
//       return o.textContent.toLowerCase().includes(ft);
//     });

//     if (!visible.length) {
//       const empty = document.createElement("div");
//       empty.className = "ss-empty";
//       empty.textContent = "No matches";
//       list.appendChild(empty);
//       return;
//     }

//     for (const o of visible) {
//       const item = document.createElement("div");
//       item.className = "ss-item";
//       item.textContent = (o.value === "" ? "— Clear —" : o.textContent);
//       item.onclick = () => {
//         selectEl.value = o.value;
//         selectEl.dispatchEvent(new Event("change"));
//         input.value = (o.value === "" ? "" : o.textContent);
//         close();
//       };
//       list.appendChild(item);
//     }
//   }

//   // Initialize input to current selection
//   input.value = selectedOption() && selectedOption().value !== "" ? selectedLabel() : "";

//   input.addEventListener("focus", () => { render(input.value); open(); });
//   input.addEventListener("input", () => { render(input.value); open(); });

//   // On blur, snap back to current selected label (prevents free-text "fake" values)
//   input.addEventListener("blur", () => {
//     setTimeout(() => {
//       input.value = (selectEl.value === "" ? "" : selectedLabel());
//       close();
//     }, 150);
//   });

//   // Sync input if select changes programmatically
//   selectEl.addEventListener("change", () => {
//     input.value = (selectEl.value === "" ? "" : selectedLabel());
//   });
// }

// function initSearchableSelects(root = document) {
//   // select.addEventListener("change", () => refreshFromSelect(select));
//   // select.addEventListener("input", () => refreshFromSelect(select));
//   root.querySelectorAll("select.searchable").forEach(sel =>
//     makeSelectSearchable(sel, { allowClear: true })
//   );
// }