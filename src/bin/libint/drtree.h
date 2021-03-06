/*
 *  This file is a part of Libint.
 *  Copyright (C) 2004-2014 Edward F. Valeev
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 */

#ifndef _libint2_src_bin_libint_drtree_h_
#define _libint2_src_bin_libint_drtree_h_

#include <smart_ptr.h>

namespace libint2 {
  
  class DGVertex;
  
  /// This is a directed rooted tree
  class DRTree :
    public EnableSafePtrFromThis<DRTree>
  {
    public:
    typedef DRTree this_type;
    
    /// If v is not on a DRTree, make a new one using v as root
    static SafePtr<DRTree> CreateRootedAt(const SafePtr<DGVertex>& v);
    ~DRTree();
    
    /// number of vertices
    unsigned int nvertices() const { return nvertices_; }
    /// the root of the tree
    const SafePtr<DGVertex>& root() const;
    /// remove all references from vertices to the tree and vice versa
    void detach();
    /// will try to add v to this subtree. Should not be used by the user
    void add_vertex(const SafePtr<DGVertex>& v);
    /// recurively detach v from this
    void detach_from(const SafePtr<DGVertex>& v);
    
    private:
    /// Create a tree starting at root
    DRTree(const SafePtr<DGVertex>& root);
    /// grows the tree from the root. Must be called after the constructor
    void grow();
    
    unsigned int nvertices_;
    SafePtr<DGVertex> root_;
  };
  
}

#endif // ifndef

