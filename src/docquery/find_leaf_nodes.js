// https://stackoverflow.com/questions/4795473/check-visibility-of-an-object-with-javascript
function isVisible(obj) {
  return obj.offsetWidth > 0 && obj.offsetHeight > 0;
}

function findLeafNodes(node) {
  if (!isVisible(node)) {
    return [];
  }

  if (node.children.length == 0 && node.innerText.length > 0) {
    return [node];
  }

  let children = [];
  for (let i = 0; i < node.children.length; i++) {
    children.push(...findLeafNodes(node.children[i]));
  }

  const childrenText = children.reduce(
    (a, x) => a + x.innerText.replace(/\s/g, ""),
    ""
  );

  // If the childrens' text contains all of the parent's text, return
  // the children. Otherwise, return the parent. This is unlikely the
  // most performant way to do this...
  parentText = node.innerText.replace(/\s/g, "");

  if (childrenText == parentText) {
    return children;
  } else {
    return [node];
  }
}

function computeBoundingBoxes(node) {
  const vw = Math.max(
    document.documentElement.clientWidth || 0,
    window.innerWidth || 0
  );
  const vh = Math.max(
    document.documentElement.clientHeight || 0,
    window.innerHeight || 0
  );

  const leafNodes = findLeafNodes(node);
  return {
    width: vw,
    height: vh,
    word_boxes: leafNodes.map((n) => ({
      text: n.innerText,
      box: n.getBoundingClientRect().toJSON(),
    })),
  };
}
