'use strict';

import {
    HorizontalSplit,
    Block,
    TabsView,
    TableRow,
    TasksKanban,
    TeamAccountsTable,
    AccountSessionsTable,
    TeamDocumentsTable,
    TeamCategoriesTable,
    TeamArchivedTasksTable,
    TeamTasksTable,
    TaskEventsTable
} from '@src/classes';

export class TeamDetails extends HorizontalSplit {

    private tabsView: TabsView;

    constructor(public teamId: number, parent: Block) {

        super(parent);

        this.tabsView = new TabsView([
            {
                text: 'Accounts',
                event: 'accounts'
            },
            {
                text: 'Tasks',
                event: 'tasks'
            },
            {
                text: 'Archived',
                event: 'archived'
            },
            
            {
                text: 'Categories',
                event: 'categories'
            },
            {
                text: 'Documents',
                event: 'documents'
            }
        ], this.leftContainer);

        this.tabsView.addClass('light-zone');

        //============
        //TEAM ACOUNTS
        //============

        this.tabsView.on('accounts', () => {

            this.rightContainer.empty();
            this.setLeftWidth(100);
                
            new TeamAccountsTable(this.teamId, false, this.tabsView.view.empty())
            .on('select', (tableRow: TableRow) => {
                new AccountSessionsTable(tableRow.rowData.ID, this.rightContainer.empty())
                .addClass('light-zone');
            });
        });

        //==========
        //TEAM TASKS
        //==========

        this.tabsView.on('tasks', () => {

            this.rightContainer.empty();
            this.setLeftWidth(100);
                
            new TeamTasksTable(this.teamId, this.tabsView.view.empty())
            .on('select', (tableRow: TableRow) => {
                new TaskEventsTable(tableRow.rowData.id, this.rightContainer.empty())
                .addClass('light-zone');
            });
        });

        //========
        //ARCHIVED
        //========

        this.tabsView.on('archived', () => {

            this.rightContainer.empty();
            this.setLeftWidth(100);
                
            new TeamArchivedTasksTable(this.teamId, this.tabsView.view.empty())
            .on('select', (tableRow: TableRow) => {
                new TaskEventsTable(tableRow.rowData.id, this.rightContainer.empty())
                .addClass('light-zone');
            });
        });

        //===============
        //TEAM CATEGORIES
        //===============

        this.tabsView.on('categories', () => {

            this.rightContainer.empty();
            this.setLeftWidth(100);

            new TeamCategoriesTable(this.teamId, this.tabsView.view.empty());
        });

        //==============
        //TEAM DOCUMENTS
        //==============

        this.tabsView.on('documents', () => {

            this.rightContainer.empty();
            this.setLeftWidth(100);

            new TeamDocumentsTable(this.teamId, this.tabsView.view.empty());
        });
    }
}