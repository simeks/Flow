#ifndef __IMIOMICS_SUBJECT_NODE_H__
#define __IMIOMICS_SUBJECT_NODE_H__

#include <Core/Flow/FlowNode.h>

class SubjectDatabase;
class SubjectNode : public FlowNode
{
    DECLARE_OBJECT(SubjectNode, FlowNode);
public:
    SubjectNode(SubjectDatabase* db = nullptr);
    virtual ~SubjectNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;

private:
    SubjectDatabase* _subject_db;
};

class SubjectPathsNode : public FlowNode
{
    DECLARE_OBJECT(SubjectPathsNode, FlowNode);
public:
    SubjectPathsNode(SubjectDatabase* db = nullptr);
    virtual ~SubjectPathsNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;

private:
    SubjectDatabase* _subject_db;
};
#endif // __IMIOMICS_SUBJECT_NODE_H__
