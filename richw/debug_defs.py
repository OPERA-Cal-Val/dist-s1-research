
class DoublyNode:
  def __init__(self,tb):
    self.tb = tb
    self.next = None
    self.prev = None
        
def up(dtb):
  if dtb.prev:
    dtb_new = dtb.prev
  else:
    dtb_new = dtb

  sl_new = dtb_new.tb.tb_frame.f_locals
  fname_new = dtb_new.tb.tb_frame.f_code.co_filename
  lineno_new = dtb_new.tb.tb_frame.f_lineno
  return dtb_new,sl_new,fname_new,lineno_new
    
def down(dtb):
  if dtb.next:
    dtb_new = dtb.next
  else: 
    dtb_new = dtb

  sl_new = dtb_new.tb.tb_frame.f_locals
  fname_new = dtb_new.tb.tb_frame.f_code.co_filename
  lineno_new = dtb_new.tb.tb_frame.f_lineno
  return dtb_new,sl_new,fname_new,lineno_new

def to_doubly(tb):
    dtb = DoublyNode(tb)
    prev = dtb
    tb_current = tb.tb_next
    while tb_current:
      new_node = DoublyNode(tb_current)
      prev.next = new_node
      new_node.prev = prev
      prev = new_node
      tb_current = tb_current.tb_next
    return dtb

def locate_err_user(dtb,userstrings):
    dtb1 = dtb
    sl = dtb1.tb.tb_frame.f_locals
    while True:
      if dtb1.next:
        fname = dtb1.tb.tb_next.tb_frame.f_code.co_filename
        if any(sub in fname for sub in userstrings):
          dtb1,sl,fname,lineno = down(dtb1)
        else:
          break
      else:
        break
    fname = dtb1.tb.tb_frame.f_code.co_filename
    lineno = dtb1.tb.tb_frame.f_lineno
    return dtb1,sl,fname,lineno
