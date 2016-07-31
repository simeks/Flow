#ifndef __IMIOMICS_SUBJECT_DATABASE_H__
#define __IMIOMICS_SUBJECT_DATABASE_H__

/*
POEM:
    'Id'
    'FatPercent'
    'WatPercent'
    'BodyMask'
    'BodySFCM'
    'FilteredFat'
    'FilteredWat'
*/
struct Subject
{
    enum Study
    {
        Study_POEM
    };

    Study study;
    std::string id;
    std::map<std::string, std::string> data;
};
struct SubjectConstraint
{
    std::string fixed_id;
    std::string moving_id;
    std::map<std::string, std::string> data;
};

typedef std::shared_ptr<Subject> SubjectPtr;
typedef std::shared_ptr<SubjectConstraint> SubjectConstraintPtr;

struct sqlite3;
class SubjectDatabase
{
public:
    SubjectDatabase();
    ~SubjectDatabase();

    bool open(const std::string& db_file);
    void close();

    SubjectPtr get_subject(const std::string& id);
    SubjectConstraintPtr get_constraint(const std::string& fixed_id, const std::string& moving_id);

    /// Checks whether the connection is currently open or not.
    bool is_open() const;

private:
    sqlite3* _db;

    std::map<std::string, SubjectPtr> _subject_cache;
    std::map<std::pair<std::string, std::string>, SubjectConstraintPtr> _constraint_cache;
};

/// Helper class for SubjectDatabase, automatically closing the database connection.
/// Example usage:
/// SubjectDatabaseScope db(db_object, "C:\\subject.db");
/// if (db->is_open())
///     ...
class SubjectDatabaseScope : NonCopyable
{
public:
    SubjectDatabaseScope(SubjectDatabase& db, const std::string& file) : _db(db)
    {
        _db.open(file);
    }
    ~SubjectDatabaseScope()
    {
        _db.close();
    }

    SubjectDatabase* operator->()
    {
        return &_db;
    }

private:
    SubjectDatabase& _db;
};

#endif // __IMIOMICS_SUBJECT_DATABASE_H__
