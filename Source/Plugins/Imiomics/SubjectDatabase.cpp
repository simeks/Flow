#include <Core/Common.h>

#include "SubjectDatabase.h"

#include <sqlite3.h>

SubjectDatabase::SubjectDatabase() : _db(nullptr)
{
}
SubjectDatabase::~SubjectDatabase()
{
    close();
}
bool SubjectDatabase::open(const std::string& db_file)
{
    int rc = sqlite3_open(db_file.c_str(), &_db);
    if (rc)
    {
        console::error("Failed to open SQLite3 database '%s': %s", db_file.c_str(), sqlite3_errmsg(_db));
        close();
        return false;
    }
    return true;
}
void SubjectDatabase::close()
{
    if (_db)
    {
        sqlite3_close(_db);
        _db = nullptr;
    }
}
SubjectPtr SubjectDatabase::get_subject(const std::string& id)
{
    if (!_db)
        return nullptr;

    if (_subject_cache.find(id) != _subject_cache.end())
        return _subject_cache.at(id);

    SubjectPtr subject = std::make_shared<Subject>();
    subject->id = id;
    subject->study = Subject::Study_POEM;

    sqlite3_stmt* stmt;

    const char* sql_q = "SELECT * FROM Subjects WHERE Id=? LIMIT 1";
    sqlite3_prepare(_db, sql_q, (int)strlen(sql_q) + 1, &stmt, nullptr);

    sqlite3_bind_text(stmt, 1, id.c_str(), (int)id.length(), nullptr);

    int s = sqlite3_step(stmt);
    if (s == SQLITE_ROW) 
    {
        for (int c = 0; c < sqlite3_column_count(stmt); ++c)
        {
            if (sqlite3_column_type(stmt, c) == SQLITE_TEXT)
            {
                const char* column = sqlite3_column_name(stmt, c);
                subject->data[column] = (const char*)sqlite3_column_text(stmt, c);
            }
        }
    }
    else if (s == SQLITE_DONE) 
    {
        // Nothing found
        subject = nullptr;
    }
    else 
    {
        // Error?
        return nullptr;
    }
    
    sqlite3_finalize(stmt);

    _subject_cache[id] = subject;
    return subject;
}
SubjectConstraintPtr SubjectDatabase::get_constraint(const std::string& fixed_id, const std::string& moving_id)
{
    assert(_db);
    if (!_db)
        return nullptr;

    std::pair<std::string, std::string> constraint_id(fixed_id, moving_id);
    if (_constraint_cache.find(constraint_id) != _constraint_cache.end())
        return _constraint_cache.at(constraint_id);

    SubjectConstraintPtr constraint = std::make_shared<SubjectConstraint>();
    constraint->fixed_id = fixed_id;
    constraint->moving_id = moving_id;

    sqlite3_stmt* stmt;

    const char* sql_q = "SELECT * FROM Constraints WHERE FixedId=? AND MovingId=? LIMIT 1";
    sqlite3_prepare(_db, sql_q, (int)strlen(sql_q) + 1, &stmt, nullptr);

    sqlite3_bind_text(stmt, 1, fixed_id.c_str(), (int)fixed_id.length(), nullptr);
    sqlite3_bind_text(stmt, 1, moving_id.c_str(), (int)moving_id.length(), nullptr);

    int s = sqlite3_step(stmt);
    if (s == SQLITE_ROW)
    {
        for (int c = 0; c < sqlite3_column_count(stmt); ++c)
        {
            if (sqlite3_column_type(stmt, c) == SQLITE_TEXT)
            {
                const char* column = sqlite3_column_name(stmt, c);
                constraint->data[column] = (const char*)sqlite3_column_text(stmt, c);
            }
        }
    }
    else if (s == SQLITE_DONE)
    {
        // Nothing found
        constraint = nullptr;
    }
    else
    {
        // Error?
        return nullptr;
    }

    sqlite3_finalize(stmt);

    _constraint_cache[constraint_id] = constraint;
    return constraint;
}
bool SubjectDatabase::is_open() const
{
    return _db != nullptr;
}
