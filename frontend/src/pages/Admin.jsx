import { useState, useEffect } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

export default function Admin() {
  const { user } = useAuth()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  // Create form state
  const [newEmail, setNewEmail] = useState('')
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    loadUsers()
  }, [])

  async function loadUsers() {
    try {
      setLoading(true)
      const data = await api.getUsers()
      setUsers(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleCreate(e) {
    e.preventDefault()
    setCreating(true)
    setError('')
    try {
      await api.createUser({
        email: newEmail,
        display_name: newName || null,
      })
      setShowCreate(false)
      setNewEmail('')
      setNewName('')
      loadUsers()
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  async function toggleActive(u) {
    try {
      await api.updateUser(u.id, { is_active: !u.is_active })
      loadUsers()
    } catch (err) {
      setError(err.message)
    }
  }

  async function toggleAdmin(u) {
    try {
      await api.updateUser(u.id, { is_admin: !u.is_admin })
      loadUsers()
    } catch (err) {
      setError(err.message)
    }
  }

  if (!user?.is_admin) {
    return (
      <div className="p-6">
        <div className="card p-8 text-center">
          <p className="text-dark-400 text-sm">Admin access required</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-dark-100">User Management</h1>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-medium rounded-lg transition-colors"
        >
          {showCreate ? 'Cancel' : 'Invite User'}
        </button>
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
          <button onClick={() => setError('')} className="ml-2 text-red-300 hover:text-red-200">&times;</button>
        </div>
      )}

      {/* Invite User Form */}
      {showCreate && (
        <form onSubmit={handleCreate} className="card space-y-3">
          <h3 className="text-sm font-medium text-dark-200">Invite New User</h3>
          <p className="text-xs text-dark-500">They'll sign in with their Google account matching this email.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <input
              type="email"
              value={newEmail}
              onChange={e => setNewEmail(e.target.value)}
              required
              placeholder="user@gmail.com"
              className="px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-sm text-dark-100 placeholder-dark-600 focus:outline-none focus:border-primary-500/50"
            />
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="Display name (optional)"
              className="px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-sm text-dark-100 placeholder-dark-600 focus:outline-none focus:border-primary-500/50"
            />
          </div>
          <button
            type="submit"
            disabled={creating}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-green-600/50 text-white text-xs font-medium rounded-lg transition-colors"
          >
            {creating ? 'Creating...' : 'Invite User'}
          </button>
        </form>
      )}

      {/* Users Table */}
      {loading ? (
        <div className="card p-8 text-center">
          <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin mx-auto" />
        </div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-dark-500 border-b border-dark-700/50">
                <th className="text-left py-2 px-3">ID</th>
                <th className="text-left py-2 px-3">Email</th>
                <th className="text-left py-2 px-3">Name</th>
                <th className="text-center py-2 px-3">Active</th>
                <th className="text-center py-2 px-3">Admin</th>
                <th className="text-left py-2 px-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id} className="border-b border-dark-700/30 hover:bg-dark-800/50">
                  <td className="py-2.5 px-3 text-dark-400 font-data">{u.id}</td>
                  <td className="py-2.5 px-3 text-dark-200">{u.email}</td>
                  <td className="py-2.5 px-3 text-dark-300">{u.display_name || '-'}</td>
                  <td className="py-2.5 px-3 text-center">
                    <button
                      onClick={() => toggleActive(u)}
                      disabled={u.id === user.id}
                      className={`w-8 h-4 rounded-full transition-colors relative ${
                        u.is_active ? 'bg-green-600' : 'bg-dark-600'
                      } ${u.id === user.id ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                        u.is_active ? 'left-4' : 'left-0.5'
                      }`} />
                    </button>
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <button
                      onClick={() => toggleAdmin(u)}
                      disabled={u.id === user.id}
                      className={`w-8 h-4 rounded-full transition-colors relative ${
                        u.is_admin ? 'bg-amber-600' : 'bg-dark-600'
                      } ${u.id === user.id ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                        u.is_admin ? 'left-4' : 'left-0.5'
                      }`} />
                    </button>
                  </td>
                  <td className="py-2.5 px-3 text-dark-500 text-xs">
                    {u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
